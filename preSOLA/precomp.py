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
    def __init__(self, data, **kwargs):
        self.data = data
        rate            = float(kwargs.pop('rate',44100))
        win_length      = int(kwargs.pop('win_length',(2048 * 3) // 2))
        time_base       = int(kwargs.pop('time_base',win_length // 2))
        time_precision  = int(kwargs.pop('time_precision',512))
        lag_base        = int(kwargs.pop('lag_base',384))
        lag_precision   = int(kwargs.pop('lag_precision',128))

        self._rate = rate
        self._win_length = win_length
        self._time_precision = time_precision
        self._time_base      = time_base
        self._time_max       = time_base + self.data.shape[0] * time_precision
        self._lag_base = lag_base
        self._lag_precision = lag_precision
        self._lag_group_count = self.data.shape[2]
        self._lag_max         = lag_base + self._lag_group_count * self._lag_precision
        self._neg_base        = 1 - self._lag_max
        self._neg_max         = 1 - self._lag_base

        self._time_precision_inv = self._time_precision ** -1
        self._lag_precision_inv = self._lag_precision ** -1

    def _time_lower_bound(self, when):
        return max(int((when - self._time_base) * self._time_precision_inv),0)

    def _time_upper_bound(self, when):
        return min(int(((when - self._time_base) * self._time_precision_inv)+1),self.data.shape[0])

    def _lag_abs_lower_bound(self, lag):
        return max(int((abs(lag) - self._lag_base) * self._lag_precision_inv),0)

    def _lag_abs_upper_bound(self, lag):
        return min(int(((abs(lag) - self._lag_base) * self._lag_precision_inv)+1),self.data.shape[2])

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
        self._program.build(options='-cl-unsafe-math-optimizations -cl-finite-math-only -cl-no-signed-zeros -cl-strict-aliasing -cl-mad-enable',devices=self._context.devices)

        rate            = float(kwargs.pop('rate',44100))
        win_len         = int(float(kwargs.pop('win_len_s',0)) * rate) or int(kwargs.pop('win_len',(2048 * 3) // 2))
        interval        = int(float(kwargs.pop('interval_s',0)) * rate) or int(kwargs.pop('interval',512))
        lag_base        = int(float(kwargs.pop('lag_base_s',0)) * rate) or int(kwargs.pop('lag_base',384))
        lag_group_size  =  int(kwargs.pop('lag_group_size', 256))
        lag_precision   = int(float(kwargs.pop('lag_precision_s',0)) * rate) or int(kwargs.pop('lag_precision',128))
        lag_max         = int(float(kwargs.pop('lag_max_s',0)) * rate) or int(kwargs.pop('lag_max',2432))
        block_size      =  int(kwargs.pop('block_size',1<<16))

        lag_group_size  = min(self._queue.device.max_work_group_size,(1 << ((lag_group_size-1).bit_length())))
        lag_step        = lag_precision // lag_group_size
        lag_precision   = lag_group_size * lag_step
        lag_group_count = (lag_max - lag_base + lag_precision - 1) // lag_precision
        lag_count       = lag_group_size * lag_group_count
        lag_max         = lag_base + lag_group_count * lag_precision

        self._rate              = rate
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

        if k2:  shape = (self._buf_size + first.shape[0] * 2, 2)
        else:   shape = (self._buf_size + first.shape[0] * 2,  )

        buf     = np.zeros(shape, dtype=np.float32)
        fill    = self._pad_pre
        used    = 0
        offset  = np.int32(self._pad_pre)
        start   = np.int32(self._win_length // 2)
        res     = []
        ibuf = None
        obuf = None


        ev0 = None
        ev1 = None
        ev_up = None
        kernel = cl.Kernel(self._program,'eval_state_2' if k2 else 'eval_state_1')#self._program.eval_state_2 if k2 else self._program.eval_state_1
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
                    lst = []
                    for b,e in res:
                        if e: [_.wait() for _ in e]
#                        p0[1].wait()
#                        p1[1].wait()
                        lst.append(b)
                    return Block(
                        data=np.concatenate(lst)
                      , rate=self._rate
                      , win_length=self._win_length
                      , time_base = self._win_length // 2
                      , time_precision = self._interval
                      , lag_base=self._lag_base
                      , lag_precision=self._lag_precision)

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


            if fill >= min(buf.shape[0],self._buf_size):
                ev_up = None
                if ibuf is None or ibuf.shape != buf.shape:
                    ibuf = cla.to_device(self._queue,buf.copy(),async=True)
                    ev_up = ibuf.events
                else:
#                    if ev0: ev0.wait()
#                    if ev1: ev1.wait()
#                    evts = []
                    ev_up = [cl.enqueue_copy(self._queue, ibuf.base_data, buf.copy(), device_offset=ibuf.offset,is_blocking=False,wait_for=ibuf.events)]
#                    ibuf.set(buf,async=True)
#                if obuf0 is None or obuf0.shape != (block_size,self._lag_group_count):
                obuf = cla.zeros(self._queue, (block_size,2, self._lag_group_count),dtype=splice_point)

                lbuf = cl.LocalMemory(self._lag_group_size * splice_point.itemsize)

#                if k2:
                ev0 = kernel(
                    self._queue
                    , (self._lag_count,2)
                    , (self._lag_group_size,1)
                    , ibuf.data
                    , obuf.data
                    , lbuf
                    , offset
                    , np.int32(start)
                    , self._win_length
                    , self._lag_base
                    , self._lag_step
                    , self._interval
                    , np.int32(block_size)
                    , wait_for = ev_up
                    )

#                o0 = np.empty(obuf0.shape,obuf0.dtype)
#                o1 = np.empty(obuf1.shape,obuf1.dtype)
                res.append((obuf.get(async=True),list(obuf.events)))
#                o0,cl.enqueue_copy(self._queue, o0, obuf0.base_data, device_offset=obuf0.offset,is_blocking=False,wait_for=list(ev0) if ev0 else None))
#                           ,(o1,cl.enqueue_copy(self._queue, o1, obuf1.base_data, device_offset=obuf1.offset,is_blocking=False,wait_for=list(ev1) if ev1 else None))))

#                if ev_up:
#                    ev_up[0].wait()
                keep = fill - block_shift
                buf[:keep] = buf[fill - keep:fill]
                buf[keep:] = 0
                fill = keep
                start += block_shift

class Searcher:
    def __init__(self, block, **kwargs):
        self.block       = block
        self._rate       = block._rate
        rate             = self._rate
        err_max         = int(float(kwargs.pop('err_max_s',0)) * rate) or int(kwargs.pop('err_max',768))
        err_precision   = int(float(kwargs.pop('err_precision_s',0))*rate) or int(kwargs.pop('err_precision', 64 ))
        lap_length      = int(float(kwargs.pop('lap_length_s',0)) * rate) or int(kwargs.pop('lap_length',1024))
        dp_step         = int(float(kwargs.pop('dp_step_s',0)) * rate) or int(kwargs.pop('dp_step',lap_length * 2))
        lookahead       = int(float(kwargs.pop('lookahead_s',0)) * rate / dp_step) or int(kwargs.pop('lookahead',6))
        solve_count     = int(kwargs.pop('solve_count',lookahead // 2))

        alpha           = float(kwargs.pop('alpha',1.0))
        delta_n         = float(kwargs.pop('delta_n',-self.block._lag_max - self.block._win_length//2))

        self._err_max           = err_max
        self._err_precision     = err_precision
        self._lap_length        = lap_length
        self._dp_step           = dp_step
        self._lookahead_count   = lookahead
        self._lookahead_distance= lookahead * dp_step
        self._solve_count       = solve_count
        self._solve_distance    = solve_count * dp_step

        self._err_precision_inv = self._err_precision ** -1
        self._err_bins_center   = np.ceil(self._err_max * self._err_precision_inv - 0.5) + 0.5
        self._err_bins_count    = self._err_bin(self._err_max) + 1

        self._t_o               = 0
        self._alpha             = alpha
        self._delta_n           = delta_n
        self._delta_i           = int(self._t_o - self._t_n())

        self._last_splice   = (self._t_o - self._lap_length // 2, self._delta_i)
        self._splices       = col.deque()
        self._solved_until  = self._t_o

        self.table    = np.zeros(shape=(self._lookahead_count + 1,self._err_bins_count),dtype=shift_point)
        self.tidx     = 0

    def _err_bin(self, err):
        _bin = int(err * self._err_precision_inv + self._err_bins_center)
        return None if (_bin < 0 or _bin >= self._err_bins_count) else _bin

    def _t_n(self, t_o=None):
        if t_o is None: t_o = self._t_o
        return self._alpha * ( t_o - self._delta_n)

    def _t_i(self, t_o=None):
        if t_o is None: t_o = self._t_o
        return t_o - self._delta_i

    def _advance_t_o(self, count):
        ret = []
        while count > 0:
            if self._splices:
                splice = self._splices[0]
                next_splice = splice[0] - self._lap_length // 2
                if next_splice < self._solved_until:
                    if next_splice < self._t_o + count:
                        self._last_splice = splice
                        self._delta_i = splice[1]
                        ret.append(splice)
                        self._splices.popleft()
                        adv = next_splice - self._t_o
                        self._t_o += adv
                        count     -= adv
                        continue

            if self._solved_until < self._t_o + count:
                adv = self._solved_until - self._t_o
                self._t_o += adv
                count     -= adv
                self._do_solve()
                continue

            self._t_o += count
            return ret

    def _error_at(self, t_o = None, delta_i = None):
        if delta_i is None:
            delta_i = self._delta_i
        if t_o is None:
            t_o = self._t_o
        return (self._alpha - 1) * t_o - self._alpha * self._delta_n + delta_i

    def _clear_table(self):
        self.tidx = 0
        self.table[::,::]['back'] = -1
        self.table[::,::]['err' ] = np.float32(np.inf)

    def _cur_tab_row(self):
        return self.table[self.tidx,::]

    def _next_tab_row(self):
        return self.table[self.tidx,::]

    def _do_solve(self):
        self._splices.clear()
        t_end = self._t_o + self._lookahead_distance
        if abs(self._error_at(self._t_o)) < self._err_max and abs(self._error_at(t_end)) < self._err_max:
            self._solved_until = self._t_o + self._solve_distance
            return
        t_l = max(self._t_o + self._lap_length // 2, self._last_splice[0] + self._lap_length)
        t_h = t_l + self._dp_step
        self._clear_table()
        err_l = self._error_at(t_o = t_l)
        bin_l = self._err_bin(err_l)

        if not bin_l:
            self._fill_next_relaxed(
        if bin_l:
            tab_row = self._cur_tab_row()
            tab_row[bin_l]['err'] = 0
            tab_row[bin_l]['pos'] = self._last_splice[0]
            tab_row[bin_l]['off'] = self._last_splice[1]
            self._fill_next(back=bin_l, t_h = t_h)
        else:
            tab_row = self._cur_tab_row()
            bin_l = 0
            tab_row[bin_l]['err'] = 0
            tab_row[bin_l]['pos'] = self._last_splice[0]
            tab_row[bin_l]['off'] = self._last_splice[1]
            if not self._fill_next_relaxed(back=bin_l, t_h = t_h):
                self._fix_jump()

        t_stop = t_h
        self.tidx += 1
        while self.tidx < self._lookahead_count:
            t_stop += self._dp_stop
            tab_row = self._cur_tab_row()
            have_next = False
            for bin_l,bin_v in enumerate(tab_row):
                if bin_v['err'] < np.inf:
                    have_next = True
                    self._fill_next(back=bin_l, t_h = t_stop)

            if not have_next:

                self.tidx -= 1

        if self.tidx == self._lookahead_count:
            bin_end = min(enumerate(self._cur_tab_row()),key=lambda x:x[1]['err'])
            splices = [(bin_end[1]['pos'],bin_end[1]['off'])]
            back = bin_end[1]['back']
            for row in self.table[-2::-1]:
                splices.append((row[back]['pos'],row[back]['off']))
                back = row[back]['back']
            last_splice = self._last_splice
            for splice in splices[::-1]:
                if splice != last_splice:
                    self._splices.append(splice)
                    last_splice = splice
            self._solved_until = self._t_o + self._solve_distance
            return
        self.tidx = 0
        try:
            self._relaxed_dp_solve(pos = t_l, off = self._delta_i)
            return
        except ValueError:
            pass
        self._fix_jump()

