
#pragma once

#include <Eigen/Dense>
#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/shared_ptr.hpp>

//#include "sphere.hpp"
//#include "clusterer.hpp"
//#include "sphericalKMeans.hpp"
#include "ddpmeans.hpp"
//#include "dpvMFmeans.hpp"
//#include "dir.hpp"
//#include "cat.hpp"

using namespace Eigen;
using std::cout;
using std::endl;

#ifdef BOOST_OLD
#define mt19937 boost::mt19937
#else
using boost::mt19937;
#endif

template<class T>
class DDPvMFMeans : public DDPMeans<T>
{
public:
  DDPvMFMeans(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx,
      T lambda, T beta, T Q, mt19937* pRndGen);
  virtual ~DDPvMFMeans();

//  void initialize(const Matrix<T,Dynamic,Dynamic>& x);

  virtual uint32_t optimisticLabelsAssign(uint32_t i0);
  virtual void updateLabelsSerial();
  virtual void updateLabels();
  virtual void updateCenters();
  virtual void nextTimeStep(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx);
  virtual void updateState(); // after converging for a single time instant
//  virtual MatrixXu mostLikelyInds(uint32_t n, 
//     Matrix<T,Dynamic,Dynamic>& deviates);
//  virtual T avgIntraClusterDeviation();
  virtual uint32_t indOfClosestCluster(int32_t i, T& sim_closest);

  virtual T silhouette();

  virtual T dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b);
  virtual T dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b);
  virtual bool closer(T a, T b);
  
protected:
//  T lambda_;
//  Sphere<T> S_;

  T beta_;
  T Q_; //TODO!

  Matrix<T,Dynamic,Dynamic> xSums_;

  virtual void removeCluster(uint32_t k);
  virtual Matrix<T,Dynamic,1> computeSum(uint32_t k);
  virtual void computeSums(void); // updates internal xSums_ 

  virtual void solveProblem1(T gamma, T age, T& phi, T& theta); 
  virtual void solveProblem2(const Matrix<T,Dynamic,1>& xSum, T zeta, T age, T w,
      T& phi, T& theta, T& eta); 

  virtual T distToUninstantiated(const Matrix<T,Dynamic,1>& x_i, uint32_t k);
  virtual void reInstantiatedOldCluster(const Matrix<T,Dynamic,1>& xSum, uint32_t k);

};
// --------------------------- impl -------------------------------------------

template<class T>
DDPvMFMeans<T>::DDPvMFMeans(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx, 
    T lambda, T beta, T Q, mt19937* pRndGen)
  : DDPMeans<T>(spx,lambda,0.,0.,pRndGen), beta_(beta), Q_(Q)
{
  this->Kprev_ = 0; // so that centers are initialized directly from sample mean
  this->psPrev_ = this->ps_;
  xSums_ = Matrix<T,Dynamic,Dynamic>::Zero(this->D_,1);

  assert(-2.0 < this->lambda_ && this->lambda_ < 0.0);

//  assert(this->D_ == spx->rows());
//  if(this->spx_.get() != spx.get()) this->spx_ = spx; // update the data
//  this->N_ = spx->cols();
//  this->z_.resize(this->N_);
//  this->z_.fill(this->UNASSIGNED);
}

template<class T>
DDPvMFMeans<T>::~DDPvMFMeans()
{}

template<class T>
uint32_t DDPvMFMeans<T>::indOfClosestCluster(int32_t i, T& sim_closest)
{
  int z_i = this->K_;
  sim_closest = this->lambda_+1.;
  T sim_k = 0.;
//  cout<<"cluster dists "<<i<<": "<< sim_closest;
  for (uint32_t k=0; k<this->K_; ++k)
  {
    if(this->Ns_(k) == 0) 
    {// cluster not instantiated yet in this timestep
      sim_k = distToUninstantiated(this->spx_->col(i), k);
    }else{ // cluster instantiated
      sim_k = dist(this->ps_.col(k), this->spx_->col(i));
    }
//    cout<<" "<<sim_k;
    if(closer(sim_k, sim_closest))
    {
      sim_closest = sim_k;
      z_i = k;
    }
  }
//  cout<<" => z_i="<<z_i<<endl;
  return z_i;
}

template<class T>
T DDPvMFMeans<T>::distToUninstantiated(const Matrix<T,Dynamic,1>& x_i, uint32_t k)
{
  assert(k<this->psPrev_.cols());

  T phi, theta, eta;
  T zeta = acos(max(static_cast<T>(-1.),min(static_cast<T>(1.0),
          (x_i.transpose()*this->psPrev_.col(k))(0))));
  solveProblem2(x_i, zeta, this->ts_[k], this->ws_[k], phi,theta,eta);

//  cout<<" phi = "<<phi<<" age="<<this->ts_[k]<<endl;

  return this->ws_[k]*(cos(theta)-1.) 
    +this->ts_[k]*beta_*(cos(phi)-1.) 
    +cos(eta) // no minus 1 here cancels with Z(tau) from the two other assignments
    +Q_*this->ts_[k];
//  return cos(theta) + this->ts_[k]*cos(phi) + cos(eta);

  // TODO: this was the old wrong way
//  T phi, theta;
//  solveProblem1( acos( max(static_cast<T>(-1.),min(static_cast<T>(1.), 
//            (this->ps_.col(k).transpose()*x_i)(0)))), this->ts_[k], phi, theta);
//  return cos(theta) + beta_*this->ts_[k]*cos(phi);
}

template<class T>
void DDPvMFMeans<T>::updateLabelsSerial()
{
// TODO not sure how to parallelize
//#pragma omp parallel for 
  for(uint32_t i=0; i<this->N_; ++i)
  {
    T sim = 0.;
    uint32_t z_i = indOfClosestCluster(i,sim);
    if(z_i == this->K_) 
    { // start a new cluster
      this->ps_.conservativeResize(this->D_,this->K_+1);
      this->Ns_.conservativeResize(this->K_+1); 
      this->ps_.col(this->K_) = this->spx_->col(i);
      this->Ns_(z_i) = 1.;
      this->K_ ++;
//      cout<<" added new cluster center at "<<this->spx_->col(i).transpose()<<endl;
    } else {
      if(this->Ns_[z_i] == 0)
      { // instantiated an old cluster
        reInstantiatedOldCluster(this->spx_->col(i), z_i);
      }
      this->Ns_(z_i) ++;
    }
    if(this->z_(i) != UNASSIGNED) this->Ns_(this->z_(i)) --;
    this->z_(i) = z_i;
  }

  this->Ns_.fill(0);
#pragma omp parallel for
  for(uint32_t k=0; k<this->K_; ++k)
    for(uint32_t i=0; i<this->N_; ++i)
      if(this->z_(i) == k)
      {
        this->Ns_(k) ++; 
      }
  cout<<" Ns = "<<this->Ns_.transpose()<<endl;
};

template<class T>
uint32_t DDPvMFMeans<T>::optimisticLabelsAssign(uint32_t i0)
{
  uint32_t idAction = UNASSIGNED;
#pragma omp parallel for 
  for(uint32_t i=i0; i<this->N_; ++i)
  {
    T sim = 0.;
    uint32_t z_i = indOfClosestCluster(i,sim);
#pragma omp critical
    {
      if(z_i == this->K_ || this->Ns_[z_i] == 0) 
      { // note this as starting position
        if(idAction > i) idAction = i;
      }
    }
    //    if(this->z_(i) != UNASSIGNED) this->Ns_(this->z_(i)) --;
    this->z_(i) = z_i;
  }
  return idAction;
};

template<class T>
void DDPvMFMeans<T>::updateLabels()
{
  uint32_t idAction = UNASSIGNED;
  uint32_t i0 = 0;
//  cout<<"::updateLabelsParallel"<<endl;
  do{
    idAction = optimisticLabelsAssign(i0);
//  cout<<"::updateLabelsParallel:  idAction: "<<idAction<<endl;
    if(idAction != UNASSIGNED)
    {
      T sim = 0.;
      uint32_t z_i = this->indOfClosestCluster(idAction,sim);
      if(z_i == this->K_) 
      { // start a new cluster
        this->ps_.conservativeResize(this->D_,this->K_+1);
        this->Ns_.conservativeResize(this->K_+1); 
        this->ps_.col(this->K_) = this->spx_->col(idAction);
        this->Ns_(z_i) = 1.;
        this->K_ ++;
      } 
      else if(this->Ns_[z_i] == 0)
      { // instantiated an old cluster
        reInstantiatedOldCluster(this->spx_->col(idAction), z_i);
        this->Ns_(z_i) = 1.; // set Ns of revived cluster to 1 tosignal
        // computeLabelsGPU to use the cluster;
      }
      i0 = idAction;
    }
    cout<<" K="<<this->K_<<" Ns="<<this->Ns_.transpose()<<endl;
  }while(idAction != UNASSIGNED);
//  cout<<"ps = "<<this->ps_<<endl;

  // TODO: this cost only works for a single time slice
  T cost =  0.0;
  for(uint32_t k=0; k<this->K_; ++k)
    if(this->Ns_(k) == 1.) cost += this->lambda_;

  //TODO get counts from GPU
  this->Ns_.fill(0);
#pragma omp parallel for reduction(+:cost)
  for(uint32_t k=0; k<this->K_; ++k)
    for(uint32_t i=0; i<this->N_; ++i)
      if(this->z_(i) == k)
      {
        this->Ns_(k) ++; 
        T sim_closest = dist(this->ps_.col(k), this->spx_->col(i));
        cost += sim_closest;
      }
  this->prevCost_ = this->cost_;
  this->cost_ = cost;
};

template<class T>
void DDPvMFMeans<T>::reInstantiatedOldCluster(const Matrix<T,Dynamic,1>& xSum, uint32_t k)
{
//  cout<<"xSum: "<<xSum.transpose()<<endl;
  T phi, theta, eta;
  T zeta = acos(max(static_cast<T>(-1.),min(static_cast<T>(1.0),
          (xSum.transpose()*this->psPrev_.col(k))(0)/xSum.norm())));
  solveProblem2(xSum , zeta, this->ts_[k], this->ws_[k], phi,theta,eta);

  // rotate point from mean_k towards previous mean by angle eta?
  this->ps_.col(k) = rotationFromAtoB<T>(xSum/xSum.norm(), 
      this->psPrev_.col(k), eta/(phi*this->ts_[k]+theta+eta)) 
    * xSum/xSum.norm(); 
};

template<class T>
void DDPvMFMeans<T>::computeSums(void)
{
  xSums_ = Matrix<T,Dynamic,Dynamic>::Zero(this->D_, this->K_);
#pragma omp parallel for
  for(uint32_t k=0; k<this->K_; ++k)
    for(uint32_t i=0; i<this->N_; ++i)
      if(this->z_(i) == k)
      {
        xSums_.col(k) += this->spx_->col(i); 
      }
}


template<class T>
Matrix<T,Dynamic,1> DDPvMFMeans<T>::computeSum(uint32_t k)
{
  Matrix<T,Dynamic,1> mean_k(this->D_);
  mean_k.setZero(this->D_);
  for(uint32_t i=0; i<this->N_; ++i)
    if(this->z_(i) == k)
    {
      mean_k += this->spx_->col(i); 
    }
  return mean_k;
}

template<class T>
void DDPvMFMeans<T>::updateCenters()
{
//  xSums_ = computeSums();
  computeSums();
//#pragma omp parallel for 
  for(uint32_t k=0; k<this->K_; ++k)
  {
//    Matrix<T,Dynamic,1> mean_k = this->computeCenter(k);
    if (this->Ns_(k) > 0) 
    { // have data to update kth cluster
      if(k < this->Kprev_)
      { //TODO
        reInstantiatedOldCluster(xSums_.col(k), k);
      }else{
        this->ps_.col(k)= xSums_.col(k)/xSums_.col(k).norm(); //mean_k;
      }
    }
//  cout<<this->ps_<<endl;
    assert(this->ps_(0,k) == this->ps_(0,k));
  }
};

template<class T>
void DDPvMFMeans<T>::nextTimeStep(const shared_ptr<Matrix<T,Dynamic,Dynamic> >& spx)
{
  this->psPrev_ = this->ps_;
  this->Kprev_ = this->K_;
  assert(this->D_ == spx->rows());
  if(this->spx_.get() != spx.get()) this->spx_ = spx; // update the data
  this->N_ = spx->cols();
  this->z_.resize(this->N_);
  this->z_.fill(UNASSIGNED);
};

template<class T>
void DDPvMFMeans<T>::updateState()
{
//  xSums_ = computeSums(); // already computed from last updateCenters (and no
//  label changes since)

  vector<bool> toRemove(this->K_,false);
  for(uint32_t k=0; k<this->K_; ++k)
  {
    if (k<this->ws_.size() && this->Ns_(k) > 0)
    { // instantiated cluster from previous time; 
      T phi, theta, eta;
      T zeta = acos(max(static_cast<T>(-1.),min(static_cast<T>(1.),
              (xSums_.col(k).transpose()*this->ps_.col(k))(0) 
              / xSums_.col(k).norm())));
      solveProblem2(xSums_.col(k), zeta, this->ts_[k], this->ws_[k], phi,theta,eta);
      this->ws_[k] = this->ws_[k]*cos(theta) 
        + beta_*this->ts_[k]*cos(phi)
        + xSums_.col(k).norm()*cos(eta);

      cout<<this->ws_[k]<<" : "<<theta<<" "<<phi<<" "<<eta
        <<" "<< xSums_.col(k).norm()<<endl;

//      this->ws_[k] = this->ws_[k]*cos(theta) + beta_*this->ts_[k]*cos(phi) 
//        + xSums_.col(k).norm()*cos(eta);
//      this->ws_[k] = 1./(1./this->ws_[k] + this->ts_[k]*tau_) + this->Ns_(k);
      this->ts_[k] = 0; // re-instantiated -> age is 0
    }else if(k >= this->ws_.size()){
      // new cluster
      this->ts_.push_back(0);
      //TODO
      this->ws_.push_back(xSums_.col(k).norm());//this->Ns_(k));
    }
    if(this->ts_[k]*Q_<this->lambda_) toRemove[k] = true;

    assert(this->ws_[k] == this->ws_[k]);
    cout<<"cluster "<<k
      <<"\tN="<<this->Ns_(k)
      <<"\tage="<<this->ts_[k]
      <<"\tdead? "<<this->ts_[k]*Q_<<" < "<<(this->lambda_)<<" => "<<(this->ts_[k]*Q_<this->lambda_)
      <<"\tweight="<<this->ws_[k]
      <<"\tcenter: "<<this->ps_.col(k).transpose()<<endl;

    this->ts_[k] ++; // increment all ages
  } 
  uint32_t nRemoved = 0;
  for(int32_t k=this->K_; k>=0; --k)
    if(toRemove[k])
    {
      removeCluster(k);
      nRemoved ++;
    }
  this->K_ -= nRemoved;
};


template<class T>
void DDPvMFMeans<T>::removeCluster(uint32_t k)
{
  cout<<" removeCluster "<<k<<endl;
//  for(uint32_t k=0; k<this->ws_.size(); ++k)
//    cout<<this->ws_[k]<<endl;
  //this->ws_;
  this->ws_.erase(this->ws_.begin()+k);
//  for(uint32_t k=0; k<this->ws_.size(); ++k)
//    cout<<this->ws_[k]<<endl;
  //this->ts_;
  this->ts_.erase(this->ts_.begin()+k);
  //this->Ns_;
//  cout<<"Ns before remove "<<this->Ns_.transpose()<<endl;
  this->Ns_.middleRows(k,this->Ns_.rows()-k-1) = this->Ns_.bottomRows(this->Ns_.rows()-k-1);
  this->Ns_.conservativeResize(this->Ns_.rows()-1);
//  cout<<"Ns after remove "<<this->Ns_.transpose()<<endl;
  //this->ps_;
//  cout<<this->ps_<<endl;
  this->ps_.middleCols(k,this->ps_.cols()-k-1) = this->ps_.rightCols(this->ps_.cols()-k-1);
  this->ps_.conservativeResize(this->ps_.rows(),this->ps_.cols()-1);

  this->psPrev_.middleCols(k,this->psPrev_.cols()-k-1) = this->psPrev_.rightCols(this->psPrev_.cols()-k-1);
  this->psPrev_.conservativeResize(this->psPrev_.rows(),this->psPrev_.cols()-1);
//  cout<<this->ps_<<endl;
  //this->xSums_;
//  cout<<this->xSums_<<endl;
  this->xSums_.middleCols(k,this->xSums_.cols()-k-1) = this->xSums_.rightCols(this->xSums_.cols()-k-1);
  this->xSums_.conservativeResize(this->xSums_.rows(),this->xSums_.cols()-1);
//  cout<<this->xSums_<<endl;
}


template<class T>
T DDPvMFMeans<T>::dist(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
{
//  return acos(min(1.0,max(-1.0,(a.transpose()*b)(0)))); // angular similarity
  return a.transpose()*b; // cosine similarity 
};

template<class T>
bool DDPvMFMeans<T>::closer(T a, T b)
{
//  return a<b; // if dist a is greater than dist b a is closer than b (angular dist)
  return a>b; // if dist a is greater than dist b a is closer than b (cosine dist)
};


template<class T>
T DDPvMFMeans<T>::dissimilarity(const Matrix<T,Dynamic,1>& a, const Matrix<T,Dynamic,1>& b)
{
  return acos(min(static_cast<T>(1.0),max(static_cast<T>(-1.0),static_cast<T>((a.transpose()*b)(0))))); // angular similarity
//  return a.transpose()*b; // cosine similarity 
};

template<class T>
T DDPvMFMeans<T>::silhouette()
{ 
  if(this->K_<2) return -1.0;
  cout<<"Ns "<<this->Ns_.transpose()<< " "<<this->Ns_.sum()<<" "<<this->N_<<endl;
  assert(this->Ns_.sum() == this->N_);
  computeSums();//TODO maybe unnecessary
  Matrix<T,Dynamic,1> sil(this->N_);
#pragma omp parallel for
  for(uint32_t i=0; i<this->N_; ++i)
  {
    Matrix<T,Dynamic,1> b = Matrix<T,Dynamic,1>::Zero(this->K_);
    // use 1 - dot(a,b) as dissimilarity measure (needs to be positive and larger for more dissimilar)
    for(uint32_t k=0; k<this->K_; ++k)
      if (k == this->z_(i))
        b(k) =1. -(this->spx_->col(i).transpose()*(xSums_.col(k) - this->spx_->col(i)))(0)/static_cast<T>(this->Ns_(k));
      else
        b(k) = 1. -(this->spx_->col(i).transpose()*xSums_.col(k))(0)/static_cast<T>(this->Ns_(k));
    T a_i = b(this->z_(i)); // average dist to own cluster
    T b_i = this->z_(i)==0 ? b(1) : b(0); // avg dist do closest other cluster
    for(uint32_t k=0; k<this->K_; ++k)
      if(k != this->z_(i) && b(k) == b(k) && b(k) < b_i && this->Ns_(k) > 0)
      {
        b_i = b(k);
      }
    if(a_i < b_i)
      sil(i) = 1.- a_i/b_i;
    else if(a_i > b_i)
      sil(i) = b_i/a_i - 1.;
    else
      sil(i) = 0.;
    if(sil(i) <-1 || sil(i) > 1)
      cout<<"sil. out of bounds "<<sil(i)<< " a="<<a_i<<" b="<<b_i<<endl;
  }
  return sil.sum()/static_cast<T>(this->N_);
};


template<class T>
void DDPvMFMeans<T>::solveProblem1(T gamma, T age, T& phi, T& theta)
{
  // solves
  // (1)  sin(phi) beta = sin(theta)
  // (2)  gamma = T phi + theta
  // for phi and theta
  phi = 0.0; 

  for (uint32_t i=0; i< 10; ++i)
  {
    T sinPhi = sin(phi);
    T f = - gamma + age*phi + asin(beta_*sinPhi);
    // mathematica
    T df = age + (beta_*cos(phi))/sqrt(1.-beta_*beta_*sinPhi*sinPhi); 
    T dPhi = f/df;
    phi = phi - dPhi; // Newton iteration
//    cout<<"@i="<<i<<": "<<phi<<"\t"<<dPhi<<endl;
    if(fabs(dPhi) < 1e-6) break;
  }

  theta = asin(beta_*sin(phi));
};


template<class T>
void DDPvMFMeans<T>::solveProblem2(const Matrix<T,Dynamic,1>& xSum, T zeta, 
    T age, T w, T& phi, T& theta, T& eta)
{
  // solves
  // w sin(theta) = beta sin(phi) = ||xSum||_2 sin(eta) 
  // eta + T phi + theta = zeta = acos(\mu0^T xSum/||xSum||_2)
  phi = 0.0;

//  cout<<"w="<<w<<" age="<<age<<" zeta="<<zeta<<endl;

  T L2xSum = xSum.norm();
  for (uint32_t i=0; i< 10; ++i)
  {
    T sinPhi = sin(phi);
    T cosPhi = cos(phi);
    T f = - zeta + asin(beta_/L2xSum *sinPhi) + age * phi + asin(beta_/w *sinPhi);
    T df = age + (beta_*cosPhi)/sqrt(L2xSum*L2xSum -
        beta_*beta_*sinPhi*sinPhi) + (beta_*cosPhi)/sqrt(w*w -
        beta_*beta_*sinPhi*sinPhi); 

    T dPhi = f/df;

    phi = phi - dPhi; // Newton iteration
//    cout<<"@i="<<i<<": "<<"f="<<f<<" df="<<df<<" phi="<<phi<<"\t"<<dPhi<<endl;
    if(fabs(dPhi) < 1e-6) break;
  }

  theta = asin(beta_/w *sin(phi));
  eta = asin(beta_/L2xSum *sin(phi));
};
