#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/shared_ptr.hpp>

#include <dpMMlowVar/ddpmeans.hpp>
#include <dpMMlowVar/clDataGpu.hpp>
#include <dpMMlowVar/gpuMatrix.hpp>


using namespace Eigen;
using std::cout;
using std::endl;


extern void ddpLabels_gpu( double *d_q,  double *d_p,  uint32_t *d_z, 
    uint32_t *d_Ns, double *d_ages, double *d_ws, double lambda,
    double Q, double tau, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N, uint32_t *d_iAction);
extern void ddpLabels_gpu( float *d_q,  float *d_p,  uint32_t *d_z, 
    uint32_t *d_Ns, float *d_ages, float *d_ws, float lambda, float Q, 
    float tau, uint32_t k0, uint32_t K, uint32_t i0, uint32_t N, uint32_t *d_iAction);

extern void ddpLabelsSpecial_gpu( double *d_q,  double *d_oldp, double *d_ages,
    double *d_ws, double lambda, double Q, double tau, uint32_t K, uint32_t N,
    uint32_t *d_asgnIdces);

extern void ddpLabelsSpecial_gpu( float *d_q,  float *d_oldp, float *d_ages,
    float *d_ws, float lambda, float Q, float tau, uint32_t K, uint32_t N,
    uint32_t *d_asgnIdces);

template<class T, class DS>
class DDPMeansCUDA : public DDPMeans<T,DS>
{
public:
  DDPMeansCUDA(const shared_ptr<ClDataGpu<T> >& cld,
      T lambda, T Q, T tau);
  virtual ~DDPMeansCUDA();
  
  void getZfromGpu() {this->cld_->z();};
  uint32_t* d_z(){ return this->cdl_->d_z();};
  
protected:

//  static const uint32_t MAX_UINT32 = 4294967295;

  GpuMatrix<uint32_t> d_iAction_;
  GpuMatrix<T> d_ages_;
  GpuMatrix<T> d_ws_;
  GpuMatrix<uint32_t> d_Ns_;
  GpuMatrix<T> d_p_;

  virtual uint32_t optimisticLabelsAssign(uint32_t i0);
  uint32_t computeLabelsGPU(uint32_t i0);

  virtual VectorXu initLabels();
};
// --------------------------- impl -------------------------------------------

template<class T, class DS>
DDPMeansCUDA<T,DS>::DDPMeansCUDA(const shared_ptr<ClDataGpu<T> >& cld,
      T lambda, T Q, T tau)
  : DDPMeans<T,DS>(cld,lambda,Q,tau), //d_x_(cld), d_z_(this->N_),
  d_iAction_(1), d_ages_(1), d_ws_(1), d_Ns_(1), d_p_(this->D_,1)
{}

template<class T, class DS>
DDPMeansCUDA<T,DS>::~DDPMeansCUDA()
{}

template<class T, class DS>
uint32_t DDPMeansCUDA<T,DS>::computeLabelsGPU(uint32_t i0)
{
  uint32_t iAction = UNASSIGNED;
  d_iAction_.set(iAction);
  d_Ns_.set(this->counts());
  d_ages_.set(this->ages());
  d_ws_.set(this->weights());

  // TODO not too too sure about this
  Matrix<T,Dynamic,Dynamic> ps(this->D_,this->K_);
  for(uint32_t k=0; k<this->K_; ++k)
    if(this->cls_[k]->isInstantiated())
      ps.col(k) = this->cls_[k]->centroid();
    else if(!this->cls_[k]->isInstantiated() && !this->cls_[k]->isNew())
      ps.col(k) = this->clsPrev_[k]->centroid();
  d_p_.set(ps);

//  assert(this->K_ < 17); // limitation of kernel at this point

//  cout<<"ddpvMFlabels_gpu K="<<this->K_<<endl;
//  cout<<ps<<endl;
//  d_p_.print();
//  d_ages_.print();
//  d_Ns_.print();
//  cout<<"d_Ns_ "<<d_Ns_.get().transpose()<<endl;
//  cout<<"d_ages_ "<<d_ages_.get().transpose()<<endl;

//  cout << "******************BEFORE*******************"<<endl;
  ddpLabels_gpu( this->cld_->d_x(),  d_p_.data(), this->cld_->d_z(), 
      d_Ns_.data(), d_ages_.data(), d_ws_.data(), this->cl0_.lambda(), 
      this->cl0_.Q(), this->cl0_.tau(), 0, this->K_, i0, this->N_-i0, 
      d_iAction_.data());
//  cout << "------------------AFTER--------------------"<<endl;
  d_iAction_.get(iAction); 
  return iAction;
}

template<class T, class DS>
uint32_t DDPMeansCUDA<T,DS>::optimisticLabelsAssign(uint32_t i0)
{
  return computeLabelsGPU(0); // TODO make passing i0 work!
};

template<class T,class DS>
VectorXu DDPMeansCUDA<T,DS>::initLabels()
{
  VectorXu asgnIdces = VectorXu::Ones(this->K_)*UNASSIGNED;
  GpuMatrix<uint32_t> d_asgnIdces(asgnIdces);

  ddpLabelsSpecial_gpu(this->cld_->d_x(),  d_p_.data(), 
      d_ages_.data(), d_ws_.data(), this->cl0_.lambda(), 
      this->cl0_.Q(), this->cl0_.tau(), this->K_, this->N_, d_asgnIdces.data());
  return d_asgnIdces.get();      
}
