#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <dpMMlowVar/global.hpp>

#include <dpMMlowVar/clusterer.hpp>
#include <dpMMlowVar/sphericalData.hpp>
#include <dpMMlowVar/euclideanData.hpp>

using namespace Eigen;
using std::cout;
using std::endl;

namespace dplv {

template<class T, class DS>
class KMeans : public Clusterer<T,DS>
{
public:
  KMeans(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K);
  KMeans(const shared_ptr<ClData<T> >& cld);
  virtual ~KMeans();

  virtual void updateLabels();
  virtual void updateCenters();
  virtual MatrixXu mostLikelyInds(uint32_t n, Matrix<T,Dynamic,Dynamic>& deviates);
  virtual T avgIntraClusterDeviation();

  virtual uint32_t indOfClosestCluster(int32_t i, T& sim_closest);

  virtual bool converged(T eps=1e-6) 
  {
    return this->counts().size() > 0 && this->counts().size() == prevNs_.size()
      && (prevNs_.array() == this->counts().array()).all();
  };

  virtual bool convergedCounts(uint32_t dCounts) 
  {
//    cout<<this->counts()<<endl<<prevNs_<<endl;
    if(this->counts().size() > 0 && this->counts().size() == prevNs_.size())
    {
      int dC = 0;
      for(uint32_t k=0; k<this->counts().size(); ++k)
        dC += abs(int(prevNs_(k)) - int(this->counts()(k)));
      cout<<"d counts: "<<dC<<" max: "<<dCounts<<endl;
      return dC < dCounts;
    }
    return false;
  };


protected:
  VectorXu prevNs_;
};

typedef KMeans<double, Euclidean<double> > kmeansd;
typedef KMeans<float, Euclidean<float> > kmeansf;
typedef KMeans<double, Spherical<double> > spkmd;
typedef KMeans<float, Spherical<float> > spkmf;


// --------------------------- impl -------------------------------------------
template<class T, class DS>
KMeans<T,DS>::KMeans(
    const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, uint32_t K)
  : Clusterer<T,DS>(spx,K)
{}

template<class T, class DS>
KMeans<T,DS>::KMeans( const shared_ptr<ClData<T> >& cld)
  : Clusterer<T,DS>(cld)
{}

template<class T, class DS>
KMeans<T,DS>::~KMeans()
{}

template<class T, class DS>
uint32_t KMeans<T,DS>::indOfClosestCluster(int32_t i, T& sim_closest)
{
  sim_closest = this->cls_[0]->dist(this->cld_->x()->col(i));
  uint32_t z_i = 0;
  for(uint32_t k=1; k<this->K_; ++k)
  {
    T sim_k = this->cls_[k]->dist(this->cld_->x()->col(i));
    if( DS::closer(sim_k, sim_closest))
    {
      sim_closest = sim_k;
      z_i = k;
    }
  }
  return z_i;
};

template<class T, class DS>
void KMeans<T,DS>::updateLabels()
{
  T cost = 0.;
#pragma omp parallel for reduction(+:cost)
  for(uint32_t i=0; i<this->N_; ++i)
  {
    T sim = 0;
    this->cld_->z(i) = indOfClosestCluster(i,sim);
    cost = cost + sim;
  }
  this->prevCost_ = this->cost_;
  this->cost_ = cost;
}

template<class T, class DS>
void KMeans<T,DS>::updateCenters()
{
  prevNs_.resize(this->K_);
  for(uint32_t k=0; k<this->K_; ++k)
    prevNs_(k) = this->cls_[k]->N();

  this->cld_->updateLabels(this->K_);
  this->cld_->computeSS();
  for(uint32_t k=0; k<this->K_; ++k)
    if(this->cls_[k]->N() == 0)
      this->cls_[k]->resetCenter(this->cld_);
    else
      this->cls_[k]->updateCenter(this->cld_,k);
}

template<class T, class DS>
MatrixXu KMeans<T,DS>::mostLikelyInds(uint32_t n, 
    Matrix<T,Dynamic,Dynamic>& deviates)
{
  MatrixXu inds = MatrixXu::Zero(n,this->K_);
  deviates = Matrix<T,Dynamic,Dynamic>::Ones(n,this->K_);
  deviates *= 99999.0;
  
#pragma omp parallel for 
  for (uint32_t k=0; k<this->K_; ++k)
  {
    for (uint32_t i=0; i<this->N_; ++i)
      if(this->cld_->z(i) == k)
      {
        T deviate = this->cls_[k]->dist(this->cld_->x()->col(i)); 
        for (uint32_t j=0; j<n; ++j)
          if(DS::closer(deviate, deviates(j,k)))
          {
            for(uint32_t l=n-1; l>j; --l)
            {
              deviates(l,k) = deviates(l-1,k);
              inds(l,k) = inds(l-1,k);
            }
            deviates(j,k) = deviate;
            inds(j,k) = i;
//            cout<<"after update "<<logLike<<endl;
//            Matrix<T,Dynamic,Dynamic> out(n,this->K_*2);
//            out<<logLikes.cast<T>(),inds.cast<T>();
//            cout<<out<<endl;
            break;
          }
      }
  } 
  cout<<"::mostLikelyInds: deviates"<<endl;
  cout<<deviates<<endl;
  cout<<"::mostLikelyInds: inds"<<endl;
  cout<<inds<<endl;
  return inds;
};

template<class T, class DS>
T KMeans<T,DS>::avgIntraClusterDeviation()
{
  Matrix<T,Dynamic,1> deviates(this->K_);
  deviates.setZero(this->K_);
#pragma omp parallel for 
  for (uint32_t k=0; k<this->K_; ++k)
  {
    this->cls_[k]->N() = 0.0;
    for (uint32_t i=0; i<this->N_; ++i)
      if(this->cld_->z(i) == k)
      {
        deviates(k) += this->cls_[k]->dist( this->cld_->x()->col(i));
        this->cls_[k]->N() ++;
      }
//    if(this->Ns_(k) > 0.0) deviates(k) /= this->Ns_(k);
  }
  return deviates.sum()/ this->N_;
}
}
