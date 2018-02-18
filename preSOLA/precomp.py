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
    'names':['pos','off','err','back']
  , 'formats':[np.int64,np.int64,np.float32,np.object]
    })

class Block:
    def __init__(self, data, **kwargs):
        self.data       = data
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

    def _iter_rect(self, trng, lrng):
        btrng = self._time_lower_bound(trng[0]),self._time_upper_bound(trng[1])
        if lrng[0] * lrng[1] < 0:
            raise ValueError('lag bounds must be either both non negative or both non positive',lrng)
        if lrng[0] < 0:
            blrng = self._lag_abs_lower_bound(lrng[1]),self._lag_abs_upper_bound(lrng[0])
            rect  = self.data[btrng[0]:btrng[1],1,blrng[0]:blrng[1]].flat
        else:
            blrng = self._lag_abs_lower_bound(lrng[0]),self._lag_abs_upper_bound(lrng[1])
            rect  = self.data[btrng[0]:btrng[1],0,blrng[0]:blrng[1]].flat
        t0,t1 = trng
        l0,l1 = lrng
        if l1 < l0:
            l0,l1=l1,l0
        def trim(val):
            pos,lag = val['pos'],val['lag']
            return (t0<= pos and pos <= t1 and l0 <= lag and lag <= l1)

        return filter(trim,rect)

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
        self._mem_pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self._queue))
        kernels = pathlib.Path(__file__).resolve().with_name('kernels.cl')
        with kernels.open('r') as _f:
            self._program = cl.Program(self._context, _f.read())
        self._program.build(options='-cl-no-signed-zeros -cl-strict-aliasing -cl-mad-enable',devices=self._context.devices)

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
        lbuf = cl.LocalMemory(self._lag_group_size * splice_point.itemsize)


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
                    if obuf is not None:
                        evts = list(obuf.events)
                        if evts:
                            cl.wait_for_events(evts)
                    del kernel
                    del obuf
                    del ibuf
                    del lbuf

                    return Block(
                        data=np.concatenate(res)
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
                    ibuf = cla.to_device(self._queue,buf,async=True,allocator=self._mem_pool)
                    ev_up = list(ibuf.events)
                else:
#                    ev_up = [cl.enqueue_copy(self._queue, ibuf.base_data, buf., device_offset=ibuf.offset,is_blocking=True,wait_for=ibuf.events)]
                    ibuf.set(buf,async=True)
                    ev_up = list(ibuf.events)

                if not ev_up:
                    ev_up = None

                if obuf is None or obuf.shape != (block_size,2, self._lag_group_count):
                    obuf = cla.zeros(self._queue, (block_size,2, self._lag_group_count),dtype=splice_point,allocator=self._mem_pool)
                else:
                    if obuf.events:
                        cl.wait_for_events(obuf.events)

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

                ev0.wait()
                res.append(obuf.get(async=True))
                keep = fill - block_shift
                if ev_up is not None:
                    cl.wait_for_events(ev_up)
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
        self._err_bins_count    = int(self._err_max * self._err_precision_inv + self._err_bins_center)

        self._t_o               = 0
        self._alpha             = alpha
        self._delta_n           = delta_n
        self._delta_i           = int(self._t_o - self._t_n())

        self._last_splice   = (self._t_o - self._lap_length // 2, self._delta_i, 0)
        self._splices       = col.deque()
        self._solved_until  = self._t_o

        self.table    = np.zeros(shape=(self._lookahead_count + 1,self._err_bins_count),dtype=np.object)
        self.tidx     = 0
        self._clear_table()

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
        z = np.zeros(shape=(1,),dtype=shift_point)[0]
        z['err'] = np.inf
        z['back'] = None
        self.table.ravel()[::] = [z.copy() for _ in range(np.prod(self.table.shape))]

    def _cur_tab_row(self):
        return self.table[self.tidx,::]

    def _next_tab_row(self):
        return self.table[self.tidx,::]

    def _do_solve(self):
        self._clear_table()
        self.table[0,0]['err'] = 0
        self.table[0,0]['pos'] = self._last_splice[0]
        self.table[0,0]['off'] = self._last_splice[1]
        t_l = max(self._last_splice[0] + self._lap_length, self._t_o + self._lap_length // 2)
        t_h = t_l + self._dp_step

        self._dp_populate(self.table[0,0],self.table[1],t_h)
        have_vals = False
        for idx in range(1,self.table.shape[0]-1):
            t_h += self._dp_step
            have_vals = False
            for val in self.table[idx]:
                if val['back'] is not None:
                    have_vals = True
                    self._dp_populate(val, self.table[idx+1],t_h)
            if not have_vals:
                break
        if not have_vals:
            self._clear_table()
            self.table[0,0]['err'] = 0
            self.table[0,0]['pos'] = self._last_splice[0]
            self.table[0,0]['off'] = self._last_splice[1]
            t_l = max(self._last_splice[0] + self._lap_length, self._t_o + self._lap_length // 2)
            t_h = t_l + self._dp_step

            self._relaxed_populate(self.table[0,0],self.table[1],t_h)
            for idx in range(1,self.table.shape[0]-1):
                t_h += self._dp_step
                have_vals = False
                for val in self.table[idx]:
                    if val['back'] is not None:
                        have_vals = True
                        self._dp_populate(val, self.table[idx+1],t_h)
                if not have_vals:
                    return
        e = min(self.table[-1],key=lambda x:x['err'])
        self._splices.clear()
        splices = col.deque()
        while e:
            splices.appendleft((e['pos'],e['off'],e['err']))
            e = e['back']

        splice = self._last_splice
        for s in splices:
            if s[1] != splice[1]:
                splice = (s[0],s[1],s[2]-splice[2])
                self._splices.append(splice)
        self._solved_until = self._t_o + self._solve_distance

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self,val):
        if val != self._alpha:
            self._alpha = val
            self._solved_until = self._t_o

    @property
    def delta_n(self):
        return self._delta_n

    @delta_n.setter
    def delta_n(self,val):
        if val != self._delta_n:
            self._delta_n = val
            self._solved_until = self._t_o


    def _dp_populate(self, back, row, t_h = None):
        if not back:
            return
        t_l = max(back['pos'] + self._lap_length, self._t_o + self._lap_length // 2)
        if t_h is None:
            t_h = t_l + self._dp_step

        d_i = back['off']
        _E_pre = self._E_pre(t_l,t_h,d_i)
        _E_end = self._E_end(t_l,t_h,d_i)

        aim1 = self._alpha - 1

        if aim1 < 0:
            _E_end = (max(_E_end[0],0),max(_E_end[1],0))
        else:
            _E_end = (min(_E_end[0],0),min(_E_end[1],0))

        _E_end_base = aim1 * t_h - self._alpha * self._delta_n + d_i
        def _E_end_actual(val):
            return _E_end_base + val['lag']

        e = _E_end_base
        e_bin = self._err_bin(e)
        _back_err = back['err']
        if e_bin:
            prev = row[e_bin]
            _err = _back_err
            if not prev['back'] or prev['err'] > _err:
                prev['back'] = back
                prev['pos']  = d_i
                prev['err']  = _err
                prev['off']  = d_i

        l0 = self._alpha * (self._delta_n - d_i) - self._err_max
        l1 = self._alpha * (self._delta_n - d_i) + self._err_max

        def _E_post(val):
            t_s = val['pos']
            d_s = val['lag']
            mid = aim1 * t_s + d_s
            return l0 <= mid and mid <= l1

        _rect = filter(_E_post,self.block._iter_rect(_E_pre,_E_end))

        for val in _rect:
            e = _E_end_actual(val)
            e_bin = self._err_bin(e)
            if e_bin:
                prev = row[e_bin]
                _err = val['err'] + _back_err
                if prev['back'] is None or prev['err'] > _err:
                    prev['back'] = back
                    prev['pos']  = d_i + val['pos']
                    prev['err']  = _err
                    prev['off']  = d_i + val['lag']

    def _relaxed_populate(self, back, row, t_h = None):
        print('relaxed_populate',back)
        t_l = max(back['pos'] + self._lap_length, self._t_o + self._lap_length // 2)
        if t_h is None:
            t_h = t_l + self._dp_step

        d_i = back['off']
        _E_pre = self._E_bound(t_l,t_h,d_i)
        _E_end = self._E_end(t_l,t_h,d_i)

        aim1 = self._alpha - 1

        _E_end_base = aim1 * t_h - self._alpha * self._delta_n + d_i
        def _E_end_actual(val):
            return _E_end_base + val['lag']

        e = _E_end_base
        e_bin = self._err_bin(e)
        _back_err = back['err']
        if e_bin:
            prev = row[e_bin]
            _err = _back_err
            if not prev['back'] or prev['err'] > _err:
                row[e_bin]['back'] = back
                row[e_bin]['pos']  = d_i
                row[e_bin]['err']  = _err
                row[e_bin]['off']  = d_i

        l0 = self._alpha * (self._delta_n - d_i) - self._err_max
        l1 = self._alpha * (self._delta_n - d_i) + self._err_max

        def _E_post(val):
            t_s = val['pos']
            d_s = val['lag']
            mid = aim1 * t_s + d_s
            return l0 <= mid and mid <= l1

        _rect = filter(_E_post,self.block._iter_rect(_E_pre,_E_end))

        for val in _rect:
            e = _E_end_actual(val)
            e_bin = self._err_bin(e)
            if e_bin:
                prev = row[e_bin]
                _err = val['err'] + _back_err
                if not prev['back'] or prev['err'] > _err:
                    row[e_bin]['back'] = back
                    row[e_bin]['pos']  = d_i + val['pos']
                    row[e_bin]['err']  = _err
                    row[e_bin]['off']  = d_i + val['lag']

    def _E_bound(self, t_l, t_h, d_i):
        return (t_l - d_i, t_h - d_i)

    def _E_pre(self, t_l, t_h, d_i):
        _lo = t_l - d_i
        _hi = t_h - d_i

        if self._alpha != 1.0:
            v0 = self._alpha * ( self._delta_n - d_i) - self._err_max
            v1 = self._alpha * ( self._delta_n - d_i) + self._err_max
            ai = (self._alpha - 1.0)**-1
            v0 *= ai
            v1 *= ai
            if ai < 0:
                v0,v1=v1,v0
            _lo = max(_lo,v0)
            _hi = min(_hi,v1)

        return (_lo,_hi)

    def _E_end(self, t_l, t_h, d_i):
        a  = self._alpha
        ai = 1.0 - a
        v = ai * t_h + a * self._delta_n - d_i
        return (v - self._err_max, v + self._err_max)
