import numpy as np, scipy as sp
import pyopencl as cl, pyopencl.array as cla
import itertools as it
import pathlib

splice_point = np.dtype({
    'names':['pos','lag','err']
  , 'formats':[np.int32,np.int32,np.float32]
  , 'offsets':[0,4,8]
  , 'itemsize':16}
  , align=True)

class PreContext:
    _kernel_src = str()

    def __init__(self, queue_or_context):
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
        self._program.build(options='-cl-finite-math-only -cl-denorms-are-zero',devices=self._context.devices)

        self._win_length = np.int32(2048*3//2)
        self._lag_base   = np.int32(384)
        self._lag_step   = np.int32(2)
        self._interval   = np.int32(512)
        self._nlags      = 16 * 64
    def process(self,ibuf):
        if isinstance(ibuf,np.ndarray):
            ibuf = cla.to_device(self._queue,ibuf)
        sz = ibuf.shape[0]
        k2 = False
        if len(ibuf.shape) > 1:
            if ibuf.shape[1] == 1:
                pass
            elif ibuf.shape[1] == 2:
                k2 = True
            else:
                raise ValueError('invalid dimensionality')

        max_lag = self._nlags * self._lag_step + self._lag_base
        max_pre = max_lag + self._win_length
        offset  = max_pre
        count   = sz - offset
        obuf = cla.empty(self._queue, ( (count + self._interval-1)//self._interval,self._nlags//64), dtype=splice_point)

        if k2:
            ev = self._program.eval_state_2(self._queue, ( self._nlags,),(64,),ibuf.data,obuf.data,np.int32(offset),self._win_length,self._lag_base,self._lag_step,self._interval,np.int32(count),wait_for=None)
        else:
            ev = self._program.eval_state_1(self._queue, ( self._nlags,),(64,),ibuf.data,obuf.data,np.int32(offset),self._win_length,self._lag_base,self._lag_step,self._interval,np.int32(count),wait_for=None)

        ev.wait()
        return obuf.get()


