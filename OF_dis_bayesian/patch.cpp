

#include <iostream>
#include <string>
#include <vector>

#include <thread>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include <stdio.h>  

#include "patch.h"

using std::cout;
using std::endl;
using std::vector;

namespace OFC
{
  
  typedef __v4sf v4sf;

  PatClass::PatClass(
    const camparam* cpt_in,
    const camparam* cpo_in,
    const optparam* op_in,
    const int patchid_in)
  : 
    cpt(cpt_in),
    cpo(cpo_in),
    op(op_in),
    patchid(patchid_in)
{
  pc = new patchstate;
  CreateStatusStruct(pc);

  tmp.resize(op->novals,1);
  dxx_tmp.resize(op->novals,1);
  dyy_tmp.resize(op->novals,1);


    tmp_nomeannorm.resize(op->novals,1);


      bayesian_prob = new float[1];
      memset(bayesian_prob,   0, sizeof(float)*1);
    last_level = false;
    num_valid = 0;
}

void PatClass::CreateStatusStruct(patchstate * psin)
{

  psin->pdiff.resize(op->novals,1);
  psin->pweight.resize(op->novals,1);
}

PatClass::~PatClass()
{
  delete pc;
    delete[] bayesian_prob;
}

void PatClass::InitializePatch(Eigen::Map<const Eigen::MatrixXf> * im_ao_in, Eigen::Map<const Eigen::MatrixXf> * im_ao_dx_in, Eigen::Map<const Eigen::MatrixXf> * im_ao_dy_in, const Eigen::Vector2f pt_ref_in)
{
  im_ao = im_ao_in;
  im_ao_dx = im_ao_dx_in;
  im_ao_dy = im_ao_dy_in;

  pt_ref = pt_ref_in;
  ResetPatch();

  getPatchStaticNNGrad(im_ao->data(), im_ao_dx->data(), im_ao_dy->data(), &pt_ref, &tmp, &dxx_tmp, &dyy_tmp);

  ComputeHessian();
}

void PatClass::ComputeHessian()
{
  pc->Hes(0,0) = (dxx_tmp.array() * dxx_tmp.array()).sum();
  if (pc->Hes.sum()==0)
    pc->Hes(0,0)+=1e-10;
}

void PatClass::SetTargetImage(Eigen::Map<const Eigen::MatrixXf> * im_bo_in, Eigen::Map<const Eigen::MatrixXf> * im_bo_dx_in, Eigen::Map<const Eigen::MatrixXf> * im_bo_dy_in)
{
  im_bo = im_bo_in;
  im_bo_dx = im_bo_dx_in;
  im_bo_dy = im_bo_dy_in;

  ResetPatch();
}

void PatClass::ResetPatch()
{ 
  pc->hasconverged=0; 
  pc->hasoptstarted=0; 

  pc->pt_st = pt_ref;
  pc->pt_iter = pt_ref;

  pc->p_in.setZero();
  pc->p_iter.setZero();
  pc->delta_p.setZero();    

  pc->delta_p_sqnorm = 1e-10;
  pc->delta_p_sqnorm_init = 1e-10; 
  pc->mares = 1e20;
  pc->mares_old = 1e20;
  pc->cnt=0;
  pc->invalid = false;
}



void PatClass::OptimizeStart(const Eigen::Matrix<float, 1, 1> p_in_arg)
{
  pc->p_in   = p_in_arg;
  pc->p_iter = p_in_arg;


  paramtopt();


  pc->pt_st = pc->pt_iter;


  if (pc->pt_iter[0] < cpt->tmp_lb  || pc->pt_iter[1] < cpt->tmp_lb ||
      pc->pt_iter[0] > cpt->tmp_ubw || pc->pt_iter[1] > cpt->tmp_ubh)  
  {
    pc->hasconverged=1;
    pc->pdiff = tmp;
    pc->hasoptstarted=1;
  }
  else
  {
    pc->cnt=0;
    pc->delta_p_sqnorm = 1e-10;
    pc->delta_p_sqnorm_init = 1e-10;
    pc->mares = 1e5;
    pc->mares_old = 1e20;
    pc->hasconverged=0;

    OptimizeComputeErrImg();
    
    pc->hasoptstarted=1;
    pc->invalid = false;
  }

}




void PatClass::OptimizeIter(const Eigen::Matrix<float, 1, 1> p_in_arg, const bool untilconv)

{
  if (!pc->hasoptstarted)
  {
    ResetPatch(); 
    OptimizeStart(p_in_arg);  
  }
  int oldcnt=pc->cnt;


  while (  ! (pc->hasconverged || (untilconv == false && (pc->cnt > oldcnt)))  ) 
  {
    pc->cnt++;



      pc->delta_p[0] = (dxx_tmp.array() * pc->pdiff.array()).sum();

    pc->delta_p = pc->Hes.llt().solve(pc->delta_p);
    
    pc->p_iter -= pc->delta_p;
    
    #if (SELECTMODE==2)
    if (cpt->camlr==0)
      pc->p_iter[0] = std::min(pc->p_iter[0],0.0f);
    else
      pc->p_iter[0] = std::max(pc->p_iter[0],0.0f);
    #endif
      

    paramtopt(); 
      

    if ((pc->pt_st - pc->pt_iter).norm() > op->outlierthresh
        ||                  
        pc->pt_iter[0] < cpt->tmp_lb  || pc->pt_iter[1] < cpt->tmp_lb ||
        pc->pt_iter[0] > cpt->tmp_ubw || pc->pt_iter[1] > cpt->tmp_ubh)  
    {
      pc->p_iter = pc->p_in;
      paramtopt(); 
      pc->hasconverged=1;
      pc->hasoptstarted=1;
    }
        
    OptimizeComputeErrImg();
  }


        MonteCarloConfidence(op->num_window,op->unit_disturb,bayesian_prob);












}

inline void PatClass::paramtopt()
{

      pc->pt_iter[0] = pt_ref[0] + pc->p_iter[0];

}



    bool PatClass::MonteCarloConfidence(const int num_window,
                                        const float unit_disturb,
                                        float* bayesian_prob
    ) {
        bool bool_convergence = true;
        double sigma_0_squared_2 = 8*8*2;




        getPatchStaticNNGrad(im_ao->data(), im_ao_dx->data(), im_ao_dy->data(), &pt_ref, &tmp_nomeannorm, &dxx_tmp, &dyy_tmp);
        getPatchStaticBil(im_bo->data(), &pc->pt_iter, &(pc->pdiff));
        LossComputeErrorImage_noNaN(&pc->pdiff, &pc->pweight, &pc->pdiff, &tmp_nomeannorm);




        if(num_valid<op->novals*op->ratio_patch_valid)

        {








            return false;
        }


        float mid_photometric_pat = pc->pdiff.squaredNorm() / (num_valid);
        float sum_photometric_pat = mid_photometric_pat;
        for(int i = 0;i<2*num_window+1;i++)
        {
            if(i!=num_window)
            {
                Eigen::Vector2f center = pc->pt_iter + Eigen::Vector2f(unit_disturb*(i-num_window),0);
                getPatchStaticBil(im_bo->data(), &center, &(pc->pdiff));




                LossComputeErrorImage_noNaN(&pc->pdiff, &pc->pweight, &pc->pdiff, &tmp_nomeannorm);

                if(num_valid<op->novals*op->ratio_patch_valid)
                {








                    return false;
                }

                float tmp  = pc->pdiff.squaredNorm() / (num_valid);

                sum_photometric_pat += fast_exp(-tmp/sigma_0_squared_2);


                if(tmp<mid_photometric_pat)
                {
                    bool_convergence = false;
                    break;
                }
            }
        }

        if(bool_convergence == true) {
            *bayesian_prob = fast_exp(-mid_photometric_pat / sigma_0_squared_2) / sum_photometric_pat;











        }
        return true;
    }


void PatClass::LossComputeErrorImage(Eigen::Matrix<float, Eigen::Dynamic, 1>* patdest, Eigen::Matrix<float, Eigen::Dynamic, 1>* wdest, const Eigen::Matrix<float, Eigen::Dynamic, 1>* patin,  const Eigen::Matrix<float, Eigen::Dynamic, 1>*  tmpin)
{
  v4sf * pd = (v4sf*) patdest->data(),
       * pa = (v4sf*) patin->data(),  
       * te = (v4sf*) tmpin->data(),
       * pw = (v4sf*) wdest->data();
  num_valid = 0;
  if (op->costfct==0)
  {
    for (int i=op->novals/4; i--; ++pd, ++pa, ++te, ++pw)
    {
      (*pd) = (*pa)-(*te);

        if(((*pa)[0]==0) || ((*te)[0]==0))
            (*pd)[0] = 0;
        else num_valid++;
        if(((*pa)[1]==0) || ((*te)[1]==0))
            (*pd)[1] = 0;
        else num_valid++;
        if(((*pa)[2]==0) || ((*te)[2]==0))
            (*pd)[2] = 0;
        else num_valid++;
        if(((*pa)[3]==0) || ((*te)[3]==0))
            (*pd)[3] = 0;
        else num_valid++;
      (*pw) = __builtin_ia32_andnps(op->negzero,  (*pd) );
    }
  }
  else if (op->costfct==1)
  {
    for (int i=op->novals/4; i--; ++pd, ++pa, ++te, ++pw)
    {
      (*pd) = (*pa)-(*te);
      (*pd) = __builtin_ia32_orps( __builtin_ia32_andps(op->negzero,  (*pd) )  , __builtin_ia32_sqrtps (__builtin_ia32_andnps(op->negzero,  (*pd) )) );
      (*pw) = __builtin_ia32_andnps(op->negzero,  (*pd) );
    }
  }
  else if (op->costfct==2)
  {
    for (int i=op->novals/4; i--; ++pd, ++pa, ++te, ++pw)
    {
      (*pd) = (*pa)-(*te);
      (*pd) = __builtin_ia32_orps(__builtin_ia32_andps(op->negzero,  (*pd) ), 
                                  __builtin_ia32_sqrtps (
                                    __builtin_ia32_mulps(
                                          __builtin_ia32_sqrtps (op->ones + __builtin_ia32_divps(__builtin_ia32_mulps((*pd),(*pd)) , op->normoutlier_tmpbsq)) - op->ones,
                                          op->normoutlier_tmp2bsq)
                                     )
                                    );
      (*pw) = __builtin_ia32_andnps(op->negzero,  (*pd) );                                    
    }
  }
}


    void PatClass::LossComputeErrorImage_noNaN(Eigen::Matrix<float, Eigen::Dynamic, 1>* patdest, Eigen::Matrix<float, Eigen::Dynamic, 1>* wdest, const Eigen::Matrix<float, Eigen::Dynamic, 1>* patin,  const Eigen::Matrix<float, Eigen::Dynamic, 1>*  tmpin)
    {
        v4sf * pd = (v4sf*) patdest->data(),
                * pa = (v4sf*) patin->data(),
                * te = (v4sf*) tmpin->data(),
                * pw = (v4sf*) wdest->data();
        num_valid = 0;
        if (op->costfct==0)
        {
            for (int i=op->novals/4; i--; ++pd, ++pa, ++te, ++pw)
            {
                if(((*pa)[0]==0) || ((*te)[0]==0))
                    (*pd)[0] = 0;
                else num_valid++;
                if(((*pa)[1]==0) || ((*te)[1]==0))
                    (*pd)[1] = 0;
                else num_valid++;
                if(((*pa)[2]==0) || ((*te)[2]==0))
                    (*pd)[2] = 0;
                else num_valid++;
                if(((*pa)[3]==0) || ((*te)[3]==0))
                    (*pd)[3] = 0;
                else num_valid++;
                (*pd) = (*pa)-(*te);




                (*pw) = __builtin_ia32_andnps(op->negzero,  (*pd) );
            }
        }
        else if (op->costfct==1)
        {
            for (int i=op->novals/4; i--; ++pd, ++pa, ++te, ++pw)
            {
                (*pd) = (*pa)-(*te);
                (*pd) = __builtin_ia32_orps( __builtin_ia32_andps(op->negzero,  (*pd) )  , __builtin_ia32_sqrtps (__builtin_ia32_andnps(op->negzero,  (*pd) )) );
                (*pw) = __builtin_ia32_andnps(op->negzero,  (*pd) );
            }
        }
        else if (op->costfct==2)
        {
            for (int i=op->novals/4; i--; ++pd, ++pa, ++te, ++pw)
            {
                (*pd) = (*pa)-(*te);
                (*pd) = __builtin_ia32_orps(__builtin_ia32_andps(op->negzero,  (*pd) ),
                                            __builtin_ia32_sqrtps (
                                                    __builtin_ia32_mulps(
                                                            __builtin_ia32_sqrtps (op->ones + __builtin_ia32_divps(__builtin_ia32_mulps((*pd),(*pd)) , op->normoutlier_tmpbsq)) - op->ones,
                                                            op->normoutlier_tmp2bsq)
                                            )
                );
                (*pw) = __builtin_ia32_andnps(op->negzero,  (*pd) );
            }
        }
    }

void PatClass::OptimizeComputeErrImg()
{
  getPatchStaticBil(im_bo->data(), &(pc->pt_iter), &(pc->pdiff));


  LossComputeErrorImage(&pc->pdiff, &pc->pweight, &pc->pdiff, &tmp);


  pc->delta_p_sqnorm = pc->delta_p.squaredNorm();
  if (pc->cnt==1)
    pc->delta_p_sqnorm_init = pc->delta_p_sqnorm;


  pc->mares_old = pc->mares;

  pc->mares = pc->pweight.lpNorm<1>() / (num_valid);
  if ( !  ((pc->cnt < op->max_iter) &  (pc->mares  > op->res_thresh) &  
          ((pc->cnt < op->min_iter) |  (pc->delta_p_sqnorm / pc->delta_p_sqnorm_init >= op->dp_thresh)) &
          ((pc->cnt < op->min_iter) |  (pc->mares / pc->mares_old <= op->dr_thresh)))  )
    pc->hasconverged=1;
        
}


bool PatClass::getPatchStaticNNGrad(const float* img, const float* img_dx, const float* img_dy,
                    const Eigen::Vector2f* mid_in, 
                    Eigen::Matrix<float, Eigen::Dynamic, 1>* tmp_in_e,  
                    Eigen::Matrix<float, Eigen::Dynamic, 1>*  tmp_dx_in_e, 
                    Eigen::Matrix<float, Eigen::Dynamic, 1>* tmp_dy_in_e)
{
  float *tmp_in    = tmp_in_e->data();
  float *tmp_dx_in = tmp_dx_in_e->data();
  float *tmp_dy_in = tmp_dy_in_e->data();
  
  Eigen::Vector2i pos;
  Eigen::Vector2i pos_it;
  
  pos[0] = round((*mid_in)[0]) + cpt->imgpadding;
  pos[1] = round((*mid_in)[1]) + cpt->imgpadding;
    
  int posxx = 0;

  int lb = -op->p_samp_s/2;
  int ub = op->p_samp_s/2-1;  


  int num_valid_values = 0;
  for (int j=lb; j <= ub; ++j)    
  {
    for (int i=lb; i <= ub; ++i, ++posxx)
    {
      pos_it[0] = pos[0]+i;      
      pos_it[1] = pos[1]+j;
      int idx = pos_it[0] + pos_it[1] * cpt->tmp_w;

      #if (SELECTCHANNEL==1 | SELECTCHANNEL==2)
      tmp_in[posxx] = img[idx];
      tmp_dx_in[posxx] = img_dx[idx];
      tmp_dy_in[posxx] = img_dy[idx];
      if(img[idx] != 0)
          num_valid_values++;
      #else
      idx *= 3;
      tmp_in[posxx] = img[idx]; tmp_dx_in[posxx] = img_dx[idx]; tmp_dy_in[posxx] = img_dy[idx]; ++posxx; ++idx;
      tmp_in[posxx] = img[idx]; tmp_dx_in[posxx] = img_dx[idx]; tmp_dy_in[posxx] = img_dy[idx]; ++posxx; ++idx;
      tmp_in[posxx] = img[idx]; tmp_dx_in[posxx] = img_dx[idx]; tmp_dy_in[posxx] = img_dy[idx];
      #endif
    }
  }







        float lag = tmp_in_e->sum() / num_valid_values;
        posxx = 0;
        for (int j=lb; j <= ub; ++j)
        {
            for (int i=lb; i <= ub; ++i, ++posxx)
            {
                if(tmp_in[posxx]>0)
                    tmp_in[posxx] -= lag;
            }
        }
  return true;
}


bool PatClass::getPatchStaticBil(const float* img, const Eigen::Vector2f* mid_in,  Eigen::Matrix<float, Eigen::Dynamic, 1>* tmp_in_e)
{
  float *tmp_in    = tmp_in_e->data();
  
  Eigen::Vector2f resid;
  Eigen::Vector4f we;
  Eigen::Vector4i pos;
  Eigen::Vector2i pos_it;
  

  pos[0] = ceil((*mid_in)[0]+.00001f);
  pos[1] = ceil((*mid_in)[1]+.00001f);
  pos[2] = floor((*mid_in)[0]);
  pos[3] = floor((*mid_in)[1]);  
  
  resid[0] = (*mid_in)[0] - (float)pos[2];
  resid[1] = (*mid_in)[1] - (float)pos[3];
  we[0] = resid[0]*resid[1];
  we[1] = (1-resid[0])*resid[1];
  we[2] = resid[0]*(1-resid[1]);
  we[3] = (1-resid[0])*(1-resid[1]);

  pos[0] += cpt->imgpadding;
  pos[1] += cpt->imgpadding;
  
  float * tmp_it = tmp_in;
  const float * img_a, * img_b, * img_c, * img_d, *img_e; 
   
  #if (SELECTCHANNEL==1 | SELECTCHANNEL==2)
    img_e = img    + pos[0]-op->p_samp_s/2;
  #else
    img_e = img    + (pos[0]-op->p_samp_s/2)*3;
  #endif
  
  int lb = -op->p_samp_s/2;
  int ub = op->p_samp_s/2-1;
  int num_valid_values = 0;
  for (pos_it[1]=pos[1]+lb; pos_it[1] <= pos[1]+ub; ++pos_it[1])    
  {
    #if (SELECTCHANNEL==1 | SELECTCHANNEL==2)
      img_a = img_e +  pos_it[1]    * cpt->tmp_w;
      img_c = img_e + (pos_it[1]-1) * cpt->tmp_w;
      img_b = img_a-1;
      img_d = img_c-1;
    #else
      img_a = img_e +  pos_it[1]    * cpt->tmp_w * 3;
      img_c = img_e + (pos_it[1]-1) * cpt->tmp_w * 3;
      img_b = img_a-3;
      img_d = img_c-3;
    #endif


    for (pos_it[0]=pos[0]+lb; pos_it[0] <= pos[0]+ub; ++pos_it[0], 
            ++tmp_it,++img_a,++img_b,++img_c,++img_d)    
    {
      #if (SELECTCHANNEL==1 | SELECTCHANNEL==2)
        (*tmp_it)     = we[0] * (*img_a) + we[1] * (*img_b) + we[2] * (*img_c) + we[3] * (*img_d);
        if((*img_a)==0 || (*img_b)==0 || (*img_c)==0 || (*img_d)==0)
            (*tmp_it) = 0;
        else
            num_valid_values++;
      #else
        (*tmp_it)     = we[0] * (*img_a) + we[1] * (*img_b) + we[2] * (*img_c) + we[3] * (*img_d); ++tmp_it; ++img_a; ++img_b; ++img_c; ++img_d;
        (*tmp_it)     = we[0] * (*img_a) + we[1] * (*img_b) + we[2] * (*img_c) + we[3] * (*img_d); ++tmp_it; ++img_a; ++img_b; ++img_c; ++img_d;
        (*tmp_it)     = we[0] * (*img_a) + we[1] * (*img_b) + we[2] * (*img_c) + we[3] * (*img_d);
      #endif
    }
  }





        float lag = tmp_in_e->sum() / num_valid_values;
        tmp_it = tmp_in;
        for (pos_it[1]=pos[1]+lb; pos_it[1] <= pos[1]+ub; ++pos_it[1])
        {
            for (pos_it[0]=pos[0]+lb; pos_it[0] <= pos[0]+ub; ++pos_it[0],
                    ++tmp_it)
            {
                if((*tmp_it) > 0)
                    (*tmp_it) -= lag;
            }
        }

  return true;
}  
 

}


