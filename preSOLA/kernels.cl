float2 conj(float2 a)
{
    return (float2)(a.x,-a.y);
}
float2 tran(float2 a)
{
    return a.yx;
}
float2 expa(float a)
{
    return (float2)(cospi(a),sinpi(a));
}
float2 cmul(float2 a, float2 b)
{
    return (float2)( a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

struct splice_point {
    int      pos;
    int      lag;
    float    err;
} __attribute__((aligned(16)));

__kernel void eval_state_1(
    __global const float * in
  , __global struct splice_point* out
  , __local  struct splice_point *points
  , int offset
  , int start_time
  , int win_length
  , int lag_base
  , int lag_step
  , int interval
  , int count
    )
{
    const float2 alpha = expa(2.0 / win_length);

    const int l_sz  = get_local_size(0);
    const int me    = get_global_id(0);
    const int sz    = get_global_size(0);

    const int l_me  = get_local_id(0);
    const int g_sz  = get_num_groups(0);
    const int g_me  = me / l_sz;

    const int lag = lag_base + me * lag_step;

    float ord0  = (float)(0);
    float2 ord2 = (float2)(0);


    __global const float * in_ref = in + offset + win_length;
    __global const float * out_ref = in_ref;
    __global const float * in_dif = in_ref - lag;
    __global const float * out_dif = out_ref - lag;
    int i = -win_length;
    for(; i < 0; ++i) {
        {
            const float d = in_ref[i] - in_dif[i];
            const float d2 = dot(d,d);
            ord0 += d2;
            ord2.x += d2;
            ord2 = cmul(alpha, ord2);
        }
    }
    int w = 0;
    while( i < count ) {
        struct splice_point pt = (struct splice_point) { i, lag, ord0 - ord2.x};
        points[l_me] = pt;
        for(int j = 0; j < interval && i < count; ++i,++j) {
            {
                const float d = in_ref[i] - in_dif[i];
                const float d2 = dot(d,d);
                ord0 += d2;
                ord2.x += d2;
            }
            {
                const float d = out_ref[i] - out_dif[i];
                const float d2 = dot(d,d);
                ord0   -= d2;
                ord2.x -= d2;
            }
            ord2 = cmul(alpha, ord2);
            const float err = ord0 - ord2.x;
            if(err < pt.err) {
                pt.err = err;
                pt.pos = i;
            }
        }
        points[l_me] = pt;
        work_group_barrier(CLK_GLOBAL_MEM_FENCE);
        if(l_me == 0) {
            for(int ix = 0; ix < l_sz; ++ix) {
                if(points[ix].err < pt.err)
                    pt = points[ix];
            }
            pt.pos += start_time;
            out[w * g_sz + g_me] = pt;
        }
        work_group_barrier(CLK_GLOBAL_MEM_FENCE);
        ++w;
    }
}

__kernel void eval_state_2(
    __global const float2 * in
  , __global struct splice_point* out
  , __local  struct splice_point *points
  , int offset
  , int start_time
  , int win_length
  , int lag_base
  , int lag_step
  , int interval
  , int count
    )
{
    const float2 alpha = expa(2.0 / win_length);
    const int l_sz  = get_local_size(0);
    const int me    = get_global_id(0);
    const int sz    = get_global_size(0);

    const int l_me  = get_local_id(0);
    const int g_sz  = get_num_groups(0);
    const int g_me  = me / l_sz;

    const int lag = lag_base + me * lag_step;

    float ord0  = (float)(0);
    float2 ord2 = (float2)(0);


    __global const float2 * in_ref = in + offset + win_length;
    __global const float2 * out_ref = in_ref;
    __global const float2 * in_dif = in_ref - lag;
    __global const float2 * out_dif = out_ref - lag;
    int i = -win_length;
    for(; i < 0; ++i) {
        {
            const float2 d = in_ref[i] - in_dif[i];
            const float d2 = dot(d,d);
            ord0 += d2;
            ord2.x += d2;
            ord2 = cmul(alpha, ord2);
        }
    }
    int w = 0;
    while( i < count ) {
        struct splice_point pt = (struct splice_point) { i, lag, ord0 - ord2.x};
        points[l_me] = pt;
        for(int j = 0; j < interval && i < count; ++i,++j) {
            {
                const float2 d = in_ref[i] - in_dif[i];
                const float d2 = dot(d,d);
                ord0 += d2;
                ord2.x += d2;
            }
            ord2 = cmul(alpha, ord2);
            {
                const float2 d = out_ref[i] - out_dif[i];
                const float d2 = dot(d,d);
                ord0   -= d2;
                ord2.x -= d2;
            }
            const float err = ord0 - ord2.x;
            if(err < pt.err) {
                pt.err = err;
                pt.pos = i;
            }
        }
        points[l_me] = pt;
        work_group_barrier(CLK_GLOBAL_MEM_FENCE);
        if(l_me == 0) {
            for(int ix = 0; ix < l_sz; ++ix) {
                if(points[ix].err < pt.err)
                    pt = points[ix];
            }
            pt.pos += start_time;
            out[w * g_sz + g_me] = pt;
        }
        work_group_barrier(CLK_GLOBAL_MEM_FENCE);
        ++w;
    }
}
