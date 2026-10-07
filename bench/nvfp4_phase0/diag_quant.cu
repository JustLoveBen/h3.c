// Verify the production quantize kernel output by dequantizing on host with
// the measured layout and comparing to the input.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#define CKC(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"cuda %s @%d\n",cudaGetErrorString(e_),__LINE__);return 1;} } while(0)

__device__ __forceinline__ float h3_e2m1_round(float q){float a=fabsf(q);float r;
 if(a<0.25f)r=0;else if(a<0.75f)r=.5f;else if(a<1.25f)r=1;else if(a<1.75f)r=1.5f;
 else if(a<2.5f)r=2;else if(a<3.5f)r=3;else if(a<5.f)r=4;else r=6;return copysignf(r,q);}
__device__ __forceinline__ uint8_t h3_e2m1_nibble(float q){float v=h3_e2m1_round(q);float a=fabsf(v);uint8_t m;
 if(a<.25f)m=0;else if(a<.75f)m=1;else if(a<1.25f)m=2;else if(a<1.75f)m=3;
 else if(a<2.5f)m=4;else if(a<3.5f)m=5;else if(a<5.f)m=6;else m=7;return(uint8_t)((v<0?8:0)|m);}
__device__ __forceinline__ uint8_t h3_f32_to_ue4m3(float v){int e,m,E;float f;
 if(!(v>0))return 0;f=frexpf(v,&e);E=e-1+7;
 if(E<=0){m=(int)lroundf(ldexpf(v,6)*8.f);if(m>7)m=7;return(uint8_t)m;}
 if(E>=15)return 0x7E;m=(int)lroundf((f*2.f-1.f)*8.f);if(m==8){E++;m=0;if(E>=15)return 0x7E;}return(uint8_t)((E<<3)|m);}
__device__ __forceinline__ float h3_ue4m3_to_f32(uint8_t b){int e=(b>>3)&0xF,m=b&7;
 if(e==0)return ldexpf(m/8.f,-6);if(e==0xF&&m==7)return NAN;return ldexpf(1.f+m/8.f,e-7);}
__global__ void qk(const float*__restrict__ x,uint8_t*__restrict__ x4,uint8_t*__restrict__ sf,size_t m,size_t k){
 const size_t row=(size_t)blockIdx.x+(size_t)blockIdx.y*gridDim.x;const int tid=threadIdx.x;
 const int mb=tid>>4,lane=tid&15;const size_t nc=k>>4;if(row>=m)return;
 const float*xr=x+row*k;uint8_t*xp=x4+row*(k>>1);size_t ncb=nc>>2,tb=(row>>7)*ncb,r128=row&127u;
 int nblk=(int)(nc/16);
 for(int blk=0;blk<nblk;blk++){int g=blk*16+mb;float v=xr[(size_t)g*16+lane];float am=fabsf(v);
  am=fmaxf(am,__shfl_xor_sync(~0u,am,1));am=fmaxf(am,__shfl_xor_sync(~0u,am,2));
  am=fmaxf(am,__shfl_xor_sync(~0u,am,4));am=fmaxf(am,__shfl_xor_sync(~0u,am,8));
  float sf_=(am>0)?(am/6.f):1.f;uint8_t sfb=h3_f32_to_ue4m3(sf_);sf_=h3_ue4m3_to_f32(sfb);
  uint8_t nib=h3_e2m1_nibble(v*(1.f/sf_));
  if(lane==0){size_t cc=(size_t)g&3u;sf[(tb+((size_t)g>>2))*512+r128*4+cc]=sfb;}
  uint8_t nl=(uint8_t)__shfl_down_sync(~0u,nib,1);
  if((lane&1)==0){size_t col=(size_t)g*16+lane;xp[col>>1]=(uint8_t)((nib<<4)|(nl&0xF));}}
}

static const float E2M1[8]={0,.5f,1,1.5f,2,3,4,6};
static float ue4m3(uint8_t b){int e=(b>>3)&0xF,m=b&7;if(e==0)return ldexpf(m/8.f,-6);return ldexpf(1.f+m/8.f,e-7);}
static float sf_at(const uint8_t*sf,size_t r,size_t c,size_t nc){size_t ncb=nc/4,tile=(r/128)*ncb+c/4;return ue4m3(sf[tile*512+(r%128)*4+(c%4)]);}

int main(){
  size_t m=256,k=320,nc=k/16;
  float*hx=(float*)malloc(m*k*4); srand(3);
  for(size_t i=0;i<m*k;i++)hx[i]=((rand()%2000)-1000)/250.f;
  float*dx;uint8_t*dx4,*dsf;
  CKC(cudaMalloc(&dx,m*k*4));CKC(cudaMalloc(&dx4,m*k/2));CKC(cudaMalloc(&dsf,((m+127)/128)*(nc/4)*512));
  CKC(cudaMemcpy(dx,hx,m*k*4,cudaMemcpyHostToDevice));
  qk<<<dim3(m,1,1),256>>>(dx,dx4,dsf,m,k); CKC(cudaGetLastError()); CKC(cudaDeviceSynchronize());
  uint8_t*hx4=(uint8_t*)malloc(m*k/2);uint8_t*hsf=(uint8_t*)malloc(((m+127)/128)*(nc/4)*512);
  CKC(cudaMemcpy(hx4,dx4,m*k/2,cudaMemcpyDeviceToHost));
  CKC(cudaMemcpy(hsf,dsf,((m+127)/128)*(nc/4)*512,cudaMemcpyDeviceToHost));
  double maxrel=0,sr=0,se=0; int bad=0;
  for(size_t r=0;r<m;r++)for(size_t c=0;c<nc;c++){
    float s=sf_at(hsf,r,c,nc);
    for(int j=0;j<16;j++){size_t col=c*16+j;size_t byte=(r*k+col)/2;
      uint8_t nib=(col%2==0)?((hx4[byte]>>4)&0xF):(hx4[byte]&0xF);
      float val=E2M1[nib&7]*(nib&8?-1:1)*s; float ref=hx[r*k+col];
      double e=fabs(val-ref);sr+=ref*ref;se+=e*e;
      if(e>0.5&&bad<5){printf("r=%zu c=%zu j=%d ref=%.3f dq=%.3f sf=%.4f\n",r,c,j,ref,val,s);bad++;}
    }}
  printf("quant roundtrip rel_l2=%.4f\n",sqrt(se/(sr+1e-9)));
  return 0;
}
