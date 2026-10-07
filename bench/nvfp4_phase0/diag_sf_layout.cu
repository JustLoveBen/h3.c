// Diagnose which SF swizzle layout cuBLASLt FP4 expects.
// W = all FP4 value 1.0 (nibble 2), X = all 1.0. Then Y[r,c] = sum_k 1*1*SF
// depends only on the SF values: with all SF = s, Y = k*s. So set SF to a
// single nonzero marker at logical (r*,c*) = 1.0 and the rest 0 -> Y[r,c] is
// nonzero only where the GEMM read a scale of 1.0 for that micro-block.
// By scanning which logical position lights up, we infer the layout.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#define CKC(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"cuda %s @%d\n",cudaGetErrorString(e_),__LINE__);return 1;} } while(0)
#define CKL(x) do { cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"lt %d @%d\n",(int)s_,__LINE__);return 1;} } while(0)

static size_t sf_elems(size_t rows, size_t nc){return ((rows+127)/128)*((nc+3)/4)*512;}

int run(size_t m,size_t n,size_t k){
  const size_t nc=k/16;
  uint8_t *dw4,*dx4,*dsfw,*dsfx; __nv_bfloat16 *dy;
  CKC(cudaMalloc(&dw4,n*k/2)); CKC(cudaMalloc(&dx4,m*k/2));
  CKC(cudaMalloc(&dsfw,sf_elems(n,nc))); CKC(cudaMalloc(&dsfx,sf_elems(m,nc)));
  CKC(cudaMalloc(&dy,m*n*2));
  // all FP4 = 1.0 (nibble 2): byte 0x22
  CKC(cudaMemset(dw4,0x22,n*k/2)); CKC(cudaMemset(dx4,0x22,m*k/2));
  CKC(cudaMemset(dsfw,0,sf_elems(n,nc))); CKC(cudaMemset(dsfx,0,sf_elems(m,nc)));
  // marker: put UE4M3(1.0)=0x38 at ONE physical offset of dsfx, rest 0.
  // We vary the physical offset by encoding logical (rm,cm) under a candidate
  // layout, then check which output rows got scaled.
  cublasLtHandle_t lt; CKL(cublasLtCreate(&lt));
  cublasLtMatmulDesc_t desc; CKL(cublasLtMatmulDescCreate(&desc,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;
  CKL(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
  CKL(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
  cublasLtMatmulMatrixScale_t v16=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,s32=CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
  CKL(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&v16,sizeof(v16)));
  CKL(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&v16,sizeof(v16)));
  CKL(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_C_SCALE_MODE,&s32,sizeof(s32)));
  CKL(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_D_SCALE_MODE,&s32,sizeof(s32)));
  CKL(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&dsfw,sizeof(dsfw)));
  CKL(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&dsfx,sizeof(dsfx)));
  cublasLtPointerMode_t pm=CUBLASLT_POINTER_MODE_DEVICE;
  CKL(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_POINTER_MODE,&pm,sizeof(pm)));
  cublasLtMatrixLayout_t la,lb,lc;
  CKL(cublasLtMatrixLayoutCreate(&la,CUDA_R_4F_E2M1,k,n,k));
  CKL(cublasLtMatrixLayoutCreate(&lb,CUDA_R_4F_E2M1,k,m,k));
  CKL(cublasLtMatrixLayoutCreate(&lc,CUDA_R_16BF,n,m,n));
  cublasLtMatmulPreference_t pref; CKL(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws=64<<20; void*dws; CKC(cudaMalloc(&dws,ws));
  CKL(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws)));
  cublasLtMatmulHeuristicResult_t heur[4]; int ret=0;
  CKL(cublasLtMatmulAlgoGetHeuristic(lt,desc,la,lb,lc,lc,pref,4,heur,&ret));
  if(ret<1){fprintf(stderr,"no algo\n");return 1;}
  float*dab; CKC(cudaMalloc(&dab,8)); float hab[2]={1,0};
  CKC(cudaMemcpy(dab,hab,8,cudaMemcpyHostToDevice));
  // Weights SF all 1.0 (0x38) so A side is well-defined: Y = sum_k 1*1*SFw*SFx
  CKC(cudaMemset(dsfw,0x38,sf_elems(n,nc)));
  // Probe: set ALL dsfx to 0x38 except we keep it uniform; then Y=k (constant).
  // Instead probe single-marker: set physical offset p = candidate(rm,cm).
  // Here we just set the whole dsfx to 0x38 and print a few Y values to see
  // if uniform scale -> uniform output (sanity), then do single-marker below.
  CKC(cudaMemset(dsfx,0x38,sf_elems(m,nc)));
  CKL(cublasLtMatmul(lt,desc,dab,dw4,la,dx4,lb,dab+1,dy,lc,dy,lc,&heur[0].algo,dws,ws,0));
  CKC(cudaDeviceSynchronize());
  __nv_bfloat16 *hy=(__nv_bfloat16*)malloc(m*n*2);
  CKC(cudaMemcpy(hy,dy,m*n*2,cudaMemcpyDeviceToHost));
  printf("uniform-sf: Y[0..3]=%.1f %.1f %.1f %.1f (expect ~k=%zu)\n",
    __bfloat162float(hy[0]),__bfloat162float(hy[1]),__bfloat162float(hy[2]),__bfloat162float(hy[3]),k);

  /* Single-marker probe: dsfx all 0, one physical byte = 0x38 (scale 1.0).
   * For each physical offset p in tile 0 we run a GEMM; Y[r,c] = sum over
   * micro-blocks g of 1*1*SFw*SFx[r,g]. With SFw=1 and only SFx marker=1 at
   * the (row,col) that maps to p, only that row's micro-block col contributes
   * 16 ones => Y[row, c] = 16 for all c (since every output col n sums that
   * micro-block too, all rows share W SF=1). So the lit row tells us which
   * logical row maps to physical offset p. */
  /* Column probe: hold dsfx uniform 0x38, single-marker on dsfw (weight).
   * Y[r,c] = sum_g X[r,g]*W[c,g]*SFx[r,g]*SFw[c,g]. X,W all 1, SFx=1, and
   * SFw marker=1 only at weight (row=wrow, mblock=wcol): then Y[r,c]=16 for
   * c==wrow (only that output col's weight got scaled) — but other cols got
   * SFw=0 so Y=0 there. Lit output col = weight logical row. */
  /* Exact (row, mbcol) decode on the act side. dsfx marker at p; dsfw=1.
   * Y[lit_r, c] = sum_g SFx[lit_r,g]*SFw[c,g] * (16 ones). With dsfw=1 for all
   * (c,g) and SFx marker only at (lit_r, lit_g): Y[lit_r,c] = 16 * SFw[c,lit_g]
   * = 16 for all c. So we cannot read col from Y. Instead use a NON-uniform
   * weight: set dsfw=0 except column band, no — simplest: set X data so only
   * micro-block g* has value. Put x=1 only in micro-block gsel of row lit_r.
   * Then Y[lit_r,c]=16*SFx[lit_r,gsel] lights only if marker maps to gsel.
   * We instead print the lit row AND which output value (==16 confirms 1 mb).*/
  printf("act-side (row,col) decode; X nonzero only in mblock 3 of every row:\n");
  /* zero all X, set micro-block 3 (cols 48..63) = 1.0 (0x22) for every row */
  CKC(cudaMemset(dx4, 0, m * k / 2));
  {
    uint8_t *hxp = (uint8_t *)calloc(m * k / 2, 1);
    for (size_t r = 0; r < m; r++)
      for (size_t j = 0; j < 16; j++) {
        size_t col = 3 * 16 + j; size_t byte = (r * k + col) / 2;
        if (col % 2 == 0) hxp[byte] |= 0x20; else hxp[byte] |= 0x02;
      }
    CKC(cudaMemcpy(dx4, hxp, m * k / 2, cudaMemcpyHostToDevice));
    free(hxp);
  }
  CKC(cudaMemset(dsfw, 0x38, sf_elems(n, nc))); /* weight SF all 1 */
  for (size_t p = 0; p < 512; p++) {
    CKC(cudaMemset(dsfx, 0, sf_elems(m, nc)));
    CKC(cudaMemset(dsfx + p, 0x38, 1));
    CKC(cudaMemset(dy, 0, m * n * 2));
    CKL(cublasLtMatmul(lt, desc, dab, dw4, la, dx4, lb, dab + 1, dy, lc, dy, lc,
                       &heur[0].algo, dws, ws, 0));
    CKC(cudaDeviceSynchronize());
    CKC(cudaMemcpy(hy, dy, m * n * 2, cudaMemcpyDeviceToHost));
    int lit = -1;
    for (size_t r = 0; r < m && lit < 0; r++)
      if (__bfloat162float(hy[r * n]) > 0.5f) lit = (int)r;
    /* Only lights if marker's mbcol == 3 (X only there). */
    if (lit >= 0) printf("  phys=%4zu lit_row=%d (mbcol==3 confirmed)\n", p, lit);
  }
  free(hy);
  return 0;
}
int main(){ return run(256,384,320); }
