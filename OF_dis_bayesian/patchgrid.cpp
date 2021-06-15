




#include <iostream>
#include <string>
#include <vector>
#include <valarray>

#include <thread>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>

#include <stdio.h>  

#include "patch.h"
#include "patchgrid.h"


using std::cout;
using std::endl;
using std::vector;


namespace OFC
{
    
  PatGridClass::PatGridClass(
    const camparam* cpt_in,
    const camparam* cpo_in,
    const optparam* op_in)
  : 
    cpt(cpt_in),
    cpo(cpo_in),
    op(op_in)
  {


  steps = op->steps;
  nopw = ceil( (float)cpt->width /  (float)steps );
  noph = ceil( (float)cpt->height / (float)steps );
  const int offsetw = floor((cpt->width - (nopw-1)*steps)/2);
  const int offseth = floor((cpt->height - (noph-1)*steps)/2);

  nopatches = nopw*noph;
  pt_ref.resize(nopatches);
  p_init.resize(nopatches);
  pat.reserve(nopatches);
  
  im_ao_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);
  im_ao_dx_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);
  im_ao_dy_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);

  im_bo_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);
  im_bo_dx_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);
  im_bo_dy_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);

  int patchid=0;
  for (int x = 0; x < nopw; ++x)
  {
    for (int y = 0; y < noph; ++y)
    {
      int i = x*noph + y;

      pt_ref[i][0] = x * steps + offsetw;
      pt_ref[i][1] = y * steps + offseth;
      p_init[i].setZero();
      
      pat.push_back(new OFC::PatClass(cpt, cpo, op, patchid));    
      patchid++;
    }
  }


}

PatGridClass::~PatGridClass()
{
  delete im_ao_eg;
  delete im_ao_dx_eg;
  delete im_ao_dy_eg;

  delete im_bo_eg;
  delete im_bo_dx_eg;
  delete im_bo_dy_eg;

  for (int i=0; i< nopatches; ++i)
    delete pat[i];

}

void PatGridClass::SetComplGrid(PatGridClass *cg_in)
{
  cg = cg_in;
}


void PatGridClass::InitializeGrid(const float * im_ao_in, const float * im_ao_dx_in, const float * im_ao_dy_in)
{
  im_ao = im_ao_in;
  im_ao_dx = im_ao_dx_in;
  im_ao_dy = im_ao_dy_in;
  
  new (im_ao_eg) Eigen::Map<const Eigen::MatrixXf>(im_ao,cpt->height,cpt->width);
  new (im_ao_dx_eg) Eigen::Map<const Eigen::MatrixXf>(im_ao_dx,cpt->height,cpt->width);  
  new (im_ao_dy_eg) Eigen::Map<const Eigen::MatrixXf>(im_ao_dy,cpt->height,cpt->width);  
  
  
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nopatches; ++i)
  {
    pat[i]->InitializePatch(im_ao_eg, im_ao_dx_eg, im_ao_dy_eg, pt_ref[i]);
    p_init[i].setZero();    
  }

}

void PatGridClass::SetTargetImage(const float * im_bo_in, const float * im_bo_dx_in, const float * im_bo_dy_in)
{
  im_bo = im_bo_in;
  im_bo_dx = im_bo_dx_in;
  im_bo_dy = im_bo_dy_in;
  
  new (im_bo_eg) Eigen::Map<const Eigen::MatrixXf>(im_bo,cpt->height,cpt->width);
  new (im_bo_dx_eg) Eigen::Map<const Eigen::MatrixXf>(im_bo_dx,cpt->height,cpt->width);
  new (im_bo_dy_eg) Eigen::Map<const Eigen::MatrixXf>(im_bo_dy,cpt->height,cpt->width);
  
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < nopatches; ++i)
    pat[i]->SetTargetImage(im_bo_eg, im_bo_dx_eg, im_bo_dy_eg);
  
}

void PatGridClass::Optimize(const bool first_level)
{
    #pragma omp parallel for schedule(dynamic,10)
    for (int i = 0; i < nopatches; ++i)
    {


        if(p_init[i](0) != 0 || first_level == true)
        {
            pat[i]->OptimizeIter(p_init[i], true);
            if(0)
                std::cout<<"["<<i<<"]: "<<p_init[i]<<"  "<<*(pat[i]->GetParam())<<"  "<<*(pat[i]->get_bayesian_prob())<<endl;
        }

    }
}  





















































void PatGridClass::InitializeFromCoarserOF(const float * flow_prev)
{
  #pragma omp parallel for schedule(dynamic,10)
  for (int ip = 0; ip < nopatches; ++ip)
  {
    int x = floor(pt_ref[ip][0] / 2);
    int y = floor(pt_ref[ip][1] / 2); 
    int i = y*(cpt->width/2) + x;
    

    p_init[ip](0) = flow_prev[  i  ]*2;
  }
}


    void PatGridClass::AggregateFlowDense(float *flowout) const
    {
        float* we = new float[cpt->width * cpt->height];

        memset(flowout, 0, sizeof(float) * (op->nop * cpt->width * cpt->height) );
        memset(we,      0, sizeof(float) * (          cpt->width * cpt->height) );

#ifdef USE_PARALLEL_ON_FLOWAGGR
#pragma omp parallel for schedule(static)
#endif
        for (int ip = 0; ip < nopatches; ++ip)
        {

            if (pat[ip]->IsValid())
            {

                const Eigen::Matrix<float, 1, 1>* fl = pat[ip]->GetParam();
                Eigen::Matrix<float, 1, 1> flnew;


                const float * pweight = pat[ip]->GetpWeightPtr();

                int lb = -op->p_samp_s/2;
                int ub = op->p_samp_s/2-1;

                for (int y = lb; y <= ub; ++y)
                {
                    for (int x = lb; x <= ub; ++x, ++pweight)
                    {
                        int yt = (y + pt_ref[ip][1]);
                        int xt = (x + pt_ref[ip][0]);

                        if (xt >= 0 && yt >= 0 && xt < cpt->width && yt < cpt->height)
                        {

                            int i = yt*cpt->width + xt;

#if (SELECTCHANNEL==1 | SELECTCHANNEL==2)
                            float absw = 1.0f /  (float)(std::max(op->minerrval  ,*pweight));
#else
                            float absw = (float)(std::max(op->minerrval  ,*pweight)); ++pweight;
                  absw+= (float)(std::max(op->minerrval  ,*pweight)); ++pweight;
                  absw+= (float)(std::max(op->minerrval  ,*pweight));
            absw = 1.0f / absw;
#endif

                            flnew = (*fl) * absw;
                            we[i] += absw;


                            flowout[i] += flnew[0];

                        }
                    }
                }
            }
        }


        if (cg)
        {
            Eigen::Vector4f wbil;
            Eigen::Vector4i pos;

#ifdef USE_PARALLEL_ON_FLOWAGGR
#pragma omp parallel for schedule(static)
#endif
            for (int ip = 0; ip < cg->nopatches; ++ip)
            {
                if (cg->pat[ip]->IsValid())
                {
#if (SELECTMODE==1)
                    const Eigen::Vector2f*            fl = (cg->pat[ip]->GetParam());
          Eigen::Vector2f flnew;
#else
                    const Eigen::Matrix<float, 1, 1>* fl = (cg->pat[ip]->GetParam());
                    Eigen::Matrix<float, 1, 1> flnew;
#endif

                    const Eigen::Vector2f rppos = cg->pat[ip]->GetPointPos();
                    const float * pweight = cg->pat[ip]->GetpWeightPtr();

                    Eigen::Vector2f resid;


                    pos[0] = ceil(rppos[0] +.00001);
                    pos[1] = ceil(rppos[1] +.00001);
                    pos[2] = floor(rppos[0]);
                    pos[3] = floor(rppos[1]);

                    resid[0] = rppos[0] - pos[2];
                    resid[1] = rppos[1] - pos[3];
                    wbil[0] = resid[0]*resid[1];
                    wbil[1] = (1-resid[0])*resid[1];
                    wbil[2] = resid[0]*(1-resid[1]);
                    wbil[3] = (1-resid[0])*(1-resid[1]);

                    int lb = -op->p_samp_s/2;
                    int ub = op->p_samp_s/2-1;


                    for (int y = lb; y <= ub; ++y)
                    {
                        for (int x = lb; x <= ub; ++x, ++pweight)
                        {

                            int yt = y + pos[1];
                            int xt = x + pos[0];
                            if (xt >= 1 && yt >= 1 && xt < (cpt->width-1) && yt < (cpt->height-1))
                            {

#if (SELECTCHANNEL==1 | SELECTCHANNEL==2)
                                float absw = 1.0f /  (float)(std::max(op->minerrval  ,*pweight));
#else
                                float absw = (float)(std::max(op->minerrval  ,*pweight)); ++pweight;
                      absw+= (float)(std::max(op->minerrval  ,*pweight)); ++pweight;
                      absw+= (float)(std::max(op->minerrval  ,*pweight));
                absw = 1.0f / absw;
#endif


                                flnew = (*fl) * absw;

                                int idxcc =  xt    +  yt   *cpt->width;
                                int idxfc = (xt-1) +  yt   *cpt->width;
                                int idxcf =  xt    + (yt-1)*cpt->width;
                                int idxff = (xt-1) + (yt-1)*cpt->width;

                                we[idxcc] += wbil[0] * absw;
                                we[idxfc] += wbil[1] * absw;
                                we[idxcf] += wbil[2] * absw;
                                we[idxff] += wbil[3] * absw;


                                flowout[idxcc] -= wbil[0] * flnew[0];
                                flowout[idxfc] -= wbil[1] * flnew[0];
                                flowout[idxcf] -= wbil[2] * flnew[0];
                                flowout[idxff] -= wbil[3] * flnew[0];
                            }
                        }
                    }
                }
            }
        }

#pragma omp parallel for schedule(static, 100)

        for (int yi = 0; yi < cpt->height; ++yi)
        {
            for (int xi = 0; xi < cpt->width; ++xi)
            {
                int i    = yi*cpt->width + xi;
                if (we[i]>0)
                {

                    flowout[i] /= we[i];

                }
            }
        }

        delete[] we;
    }



void PatGridClass::AggregateFlowDense(float *flowout,float *probabilityout, const bool bool_last, const float* spatial_prob) const
{
  float* we = new float[cpt->width * cpt->height];
  unsigned int* num_pat = new unsigned int[cpt->width * cpt->height];
  
  memset(flowout, 0, sizeof(float) * (op->nop * cpt->width * cpt->height) );
  memset(we,      0, sizeof(float) * (          cpt->width * cpt->height) );
  memset(num_pat, 0, sizeof(unsigned int) * (   cpt->width * cpt->height) );



  #ifdef USE_PARALLEL_ON_FLOWAGGR
    #pragma omp parallel for schedule(static)  
  #endif


















  for (int ip = 0; ip < nopatches; ++ip)
  {       
    
    if (pat[ip]->IsValid())
    {

      const Eigen::Matrix<float, 1, 1>* fl = pat[ip]->GetParam();
      Eigen::Matrix<float, 1, 1> flnew;

      


      const float * bayesian_prob = pat[ip]->get_bayesian_prob();
      int lb = -op->p_samp_s/2;
      int ub = op->p_samp_s/2-1;

      float probability  = *bayesian_prob;
      if(probability == 0)
          continue;
      for (int y = lb; y <= ub; ++y)
      {   
        for (int x = lb; x <= ub; ++x, ++bayesian_prob)
        {
          int yt = (y + pt_ref[ip][1]);
          int xt = (x + pt_ref[ip][0]);

          if (xt >= 0 && yt >= 0 && xt < cpt->width && yt < cpt->height)
          {
            int i = yt*cpt->width + xt;








              num_pat[i]++;
              int ind_in_patch = (x-lb)*op->p_samp_s+y-lb;

              flnew = (*fl) * probability * spatial_prob[ind_in_patch];
              we[i] += probability * spatial_prob[ind_in_patch];









            #if (SELECTMODE==1)
            flowout[2*i]   += flnew[0];
            flowout[2*i+1] += flnew[1];
            #else
            flowout[i] += flnew[0]; 
            #endif
          }
        }
      }
    }
  } 
  

  #pragma omp parallel for schedule(static, 100)    

  for (int yi = 0; yi < cpt->height; ++yi)
  {
    for (int xi = 0; xi < cpt->width; ++xi)
    { 
      int i    = yi*cpt->width + xi;
        if (bool_last) {
            if (we[i] > 0 && num_pat[i] > 1)
            {
                flowout[i] /= we[i];






            } else { flowout[i] = std::numeric_limits<float>::quiet_NaN();}
        } else {
            if (we[i] > 0) {
                flowout[i] /= we[i];
            } else { flowout[i] = 0; }
        }
    }
  }
  
  delete[] we;
  delete[] num_pat;
}



}


