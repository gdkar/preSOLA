import numpy as np, scipy as sp
import pyopencl as cl, pyopencl.array as cla
import itertools as it
import collections as col
import pathlib

splice_point = np.dtype({
    'names':['pos','lag','err']
  , 'formats':[np.int32,np.int32,np.float32]
  , 'offsets':[0,4,8]
  , 'itemsize':16
    }
  , align=True)

shift_point = np.dtype({
    'names':['pos','off','back','err','e_max','e_min']
  , 'formats':[np.int64,np.int64,np.int32,np.float32,np.float32,np.float32]
    }
  , align = True)

class Block:
    def __init__(self, data=None, lag_base = 384, lag_step = 2 * 64, nlags = 16, pos_base = 0, pos_step = 512):
        if data is None:
            data = np.zeros(shape=(nlags,1,),dtype=splice_point)
        self.data     = data
        self.nlags    = data.shape[1]
        self.lag_base = lag_base
        self.lag_step = lag_step
        self.pos_base = pos_base
        self.pos_step = pos_step

    def ix_guess(self, pos, lag, back = True):
        ix_lag = (lag - self.lag_base) // self.lag_step
        if not back:
            pos -= (lag + self.lag_step - 1)
        ix_pos = (pos - self.pos_base) // self.pos_step
        return (ix_pos, ix_lag)

class PreContext:
    _kernel_src = str()

    def __init__(self, queue_or_context = None, **kwargs):
        if isinstance(queue_or_context,cl.Context):
            context_obj = queue_or_context
            queue_obj   = cl.CommandQueue(context_obj)
        elif isinstance(queue_or_context, cl.CommandQueue):
            queue_obj = queue_or_context
            context_obj = queue_obj.context
        else:
            context_obj = cl.Context()
            queue_obj   = cl.CommandQueue(context_obj)
        self._context = context_obj
        self._queue   = queue_obj
        kernels = pathlib.Path(__file__).resolve().with_name('kernels.cl')
        with kernels.open('r') as _f:
            self._program = cl.Program(self._context, _f.read())
        self._program.build(options='-cl-finite-math-only',devices=self._context.devices)

        rate            = int(kwargs.pop('rate',44100))
        win_len         = (int(kwargs.pop('win_len_s',0)) * rate) or int(kwargs.pop('win_len',(2048 * 3) // 2))
        interval        = (int(kwargs.pop('interval_s',0)) * rate) or int(kwargs.pop('interval',512))
        lag_base        = (int(kwargs.pop('lag_base_s',0)) * rate) or int(kwargs.pop('lag_base',384))
        lag_group_size  =  int(kwargs.pop('lag_group_size', 64))
        lag_precision   = (int(kwargs.pop('lag_precision_s',0)) * rate) or int(kwargs.pop('lag_precision',128))
        lag_max         = (int(kwargs.pop('lag_max_s',0)) * rate) or int(kwargs.pop('lag_max',2432))
        block_size      =  int(kwargs.pop('block_size',1<<16))

        lag_group_size  = min(64,(1 << ((lag_group_size-1).bit_length())))
        lag_step        = lag_precision // lag_group_size
        lag_precision   = lag_group_size * lag_step
        lag_group_count = (lag_max - lag_base + lag_precision - 1) // lag_precision
        lag_count       = lag_group_size * lag_group_count
        lag_max         = lag_base + lag_group_count * lag_precision


        self._win_length        = np.int32(win_len)
        self._interval          = np.int32(interval)
        self._lag_base          = np.int32(lag_base)
        self._lag_step          = np.int32(lag_step)
        self._lag_group_size    = np.int32(lag_group_size)
        self._lag_group_count   = np.int32(lag_group_count)
        self._lag_count         = lag_count
        self._lag_max           = lag_max
        self._lag_precision     = lag_precision
        self._block_size        = block_size
        self._block_shift       = block_size * interval
        self._pad_pre           = lag_max
        self._pad_post          = win_len + lag_max
        self._buf_size          = self._pad_pre + self._pad_post + self._block_shift

    def process(self,it):
        if isinstance(it, np.ndarray):
            it = iter([it])

        first = next(it)

        k2 = False
        if first.ndim > 1:
            if first.shape[1] == 1: pass
            elif first.shape[1] == 2: k2 = True
            else: raise ValueError('invalid dimensionality')

        block_size = self._block_size
        block_shift= self._block_shift

        if k2:  shape = (self._buf_size, 2)
        else:   shape = (self._buf_size,)

        buf     = np.zeros(shape, dtype=np.float32)
        fill    = self._pad_pre
        used    = 0
        offset  = np.int32(self._pad_pre)
        start   = np.int32(self._win_length // 2)
        res     = []

        while True:
            if used == first.shape[0]:
                if it:
                    used = 0
                    try:
                        first = next(it)
                    except StopIteration:
                        intervals = (fill - self._pad_pre + self._interval - 1) // self._interval
                        pad = (intervals * self._interval + self._pad_pre + self._pad_post) - fill
                        print('finishing up, intervals={}, pad={}'.format(intervals,pad))
                        if k2:  first = np.zeros(shape=(pad,2),dtype=np.float32)
                        else:   first = np.zeros(shape=(pad,),dtype=np.float32)
                        print('first:',first)
                        it = None
                else:
                    return res

            if used < first.shape[0] and fill < buf.shape[0]:
                chunk   = min(buf.shape[0] - fill, first.shape[0]-used)
#                print('adding a chunk of size {}'.format(chunk))
#                print('fill is {}/{}'.format(fill,buf.shape[0]))
                buf[fill:fill + chunk] = first[used:used + chunk]
                fill += chunk
                used += chunk

            if used == first.shape[0] and fill < buf.shape[0] and not it:
                print("we're done, processing the last {} samples".format(fill))
                buf = buf[:fill]
                block_size = (fill - self._pad_pre - self._pad_post) // self._interval
                block_shift= block_size * self._interval

            if fill == buf.shape[0]:
                ibuf = cla.to_device(self._queue,buf)
                obuf0 = cla.empty(self._queue, (block_size,self._lag_group_count),dtype=splice_point)
                obuf1 = cla.empty(self._queue, (block_size,self._lag_group_count),dtype=splice_point)

                lbuf0 = cl.LocalMemory(self._lag_group_size * splice_point.itemsize)
                lbuf1 = cl.LocalMemory(self._lag_group_size * splice_point.itemsize)

                if k2:
                    ev0 = self._program.eval_state_2(
                        self._queue
                      , (self._lag_count,)
                      , (self._lag_group_size,)
                      , ibuf.data
                      , obuf0.data
                      , lbuf0
                      , offset
                      , np.int32(start)
                      , self._win_length
                      , self._lag_base
                      , self._lag_step
                      , self._interval
                      , np.int32(block_shift)
                      , wait_for = None
                        )

                    ev1 = self._program.eval_state_2(
                        self._queue
                      , (self._lag_count,)
                      , (self._lag_group_size,)
                      , ibuf.data
                      , obuf1.data
                      , lbuf1
                      , offset
                      , np.int32(start)
                      , self._win_length
                      , -self._lag_base
                      , -self._lag_step
                      , self._interval
                      , np.int32(block_shift)
                      , wait_for = None
                        )
                else:
                    ev0 = self._program.eval_state_1(
                        self._queue
                      , (self._lag_count,)
                      , (self._lag_group_size,)
                      , ibuf.data
                      , obuf0.data
                      , lbuf0
                      , offset
                      , np.int32(start)
                      , self._win_length
                      , self._lag_base
                      , self._lag_step
                      , self._interval
                      , np.int32(block_shift)
                      , wait_for = None
                        )

                    ev1 = self._program.eval_state_1(
                        self._queue
                      , (self._lag_count,)
                      , (self._lag_group_size,)
                      , ibuf.data
                      , obuf1.data
                      , lbuf1
                      , offset
                      , np.int32(start)
                      , self._win_length
                      , np.int32(-self._lag_base)
                      , np.int32(-self._lag_step)
                      , self._interval
                      , np.int32(block_shift)
                      , wait_for = None
                        )
                ev0.wait()
                ev1.wait()
                o0 = obuf0.get()
                o1 = obuf1.get()
                res.append((o0,o1))
                keep = fill - block_shift
                buf[:keep] = buf[fill - keep:fill]
                fill  -= block_shift
                start += block_shift

#        return res
#        ibuf = cla.to_device(self._queue,ibuf)
#
#        sz = ibuf.shape[0]
#
#        obuf = cla.empty(self._queue, ( (count + self._interval-1)//self._interval,self._lag_group_count), dtype=splice_point)
#        lbuf = cl.LocalMemory(self._lag_group_size * splice_point.itemsize)
#
#        if k2:
#            ev = self._program.eval_state_2(
#                self._queue
#              , ( self._lag_count,)
#              , (self._lag_group_size,)
#              , ibuf.data
#              , obuf.data
#              , lbuf
#              , np.int32(offset)
#              , np.int32(start)
#              , self._win_length
#              , self._lag_base
#              , self._lag_step
#              , self._interval
#              , np.int32(count)
#              , wait_for=None)
#        else:
#            ev = self._program.eval_state_1(
#                self._queue
#              , ( self._lag_count,)
#              , (self._lag_group_size,)
#              , ibuf.data
#              , obuf.data
#              , lbuf
#              , np.int32(offset)
#              , np.int32(start)
#              , self._win_length
#              , self._lag_base
#              , self._lag_step
#              , self._interval
#              , np.int32(count)
#              , wait_for=None)
#
#        ev.wait()
#        return obuf.get()

class Searcher:
    def __init__(self, sample_rate = 44100, block = None, err_step = 64, err_max = 1024, time_splice = 512, time_step = 1024, origin_n = 0, alpha = 1.0):
        self.block       = block
        self.sample_rate = sample_rate
        self.err_step    = err_step
        self.err_max     = err_max
        self.err_bins    = (2 * err_max) // err_step
        self.err_base    = err_max // err_step
        self.time_splice = time_splice
        self.time_step   = time_step
        self.alpha       = alpha
        self.origin_n    = origin_n
        self.origin_o    = self.alpha * self.origin_n
        self.time_o      = 0

        self.table    = np.zeros(shape=(32,self.err_bins),dtype=shift_point)
        self.tidx     = 0

    def _err_index(self, err):
        return int((err + self.err_max) / self.err_step)

    def _error(self, time_o, origin_o):
        return (1-self.alpha) * time_o - origin_o + self.alpha * self.origin_n

    def _fill_table(self, time_o, src, dst):
        time_o_next = time_o + self.time_step
        dst[::].err = np.inf
        for back,sol in enumerate(src):
            if sol.err == np.inf:
                continue
            e = self._error(time_o_next, sol.off)
            if abs(e) < self.err_max:
                ei = self._err_index(e)
                if dst[ei].err > sol.err:
                    dst[ei] = sol
                    dst[ei].back = back
            l_min = e
