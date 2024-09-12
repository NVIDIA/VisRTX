// Your texture handler structure.
struct Texture_handler : public Texture_handler_base
{
  // additional data for the texture access functions can be provided here
  int example_field;
};

extern "C" __device__ void tex_lookup_float4_2d(float result[4],
    Texture_handler const *self,
    unsigned texture_idx,
    float const coord[2],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    float const crop_u[2],
    float const crop_v[2]);

extern "C" __device__ void tex_lookup_float3_2d(float result[3],
    Texture_handler const *self,
    unsigned texture_idx,
    float const coord[2],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    float const crop_u[2],
    float const crop_v[2]);

extern "C" __device__ void tex_texel_float4_2d(float result[4],
    Texture_handler const *self,
    unsigned texture_idx,
    int const coord[2]);

extern "C" __device__ void tex_lookup_float4_3d(float result[4],
    Texture_handler const *self,
    unsigned texture_idx,
    float const coord[3],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    Tex_wrap_mode wrap_w,
    float const crop_u[2],
    float const crop_v[2],
    float const crop_w[2]);

extern "C" __device__ void tex_lookup_float3_3d(float result[3],
    Texture_handler const *self,
    unsigned texture_idx,
    float const coord[3],
    Tex_wrap_mode wrap_u,
    Tex_wrap_mode wrap_v,
    Tex_wrap_mode wrap_w,
    float const crop_u[2],
    float const crop_v[2],
    float const crop_w[2]);

extern "C" __device__ void tex_texel_float4_3d(float result[4],
    Texture_handler const *self,
    unsigned texture_idx,
    int const coord[3]);

extern "C" __device__ void tex_lookup_float4_cube(float result[4],
    Texture_handler const *self,
    unsigned texture_idx,
    float const coord[3]);

extern "C" __device__ void tex_lookup_float3_cube(float result[3],
    Texture_handler const *self,
    unsigned texture_idx,
    float const coord[3]);

extern "C" __device__ void tex_resolution_2d(int result[2],
    Texture_handler const *self,
    unsigned texture_idx,
    int const uv_tile[2]);