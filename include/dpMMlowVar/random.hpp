/* Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#ifndef __RANDOM_HPP
#include <random>
#include <dpMMlowVar/timer.hpp>

//singleton design pattern for random stuff
//singletons ensure that all randomness in dynamic means
//comes from a single source 
//helps with repeatability during experiments
namespace dmeans{
class RNG{
  public:
    static std::mt19937& get(){
      static RNG instance;
      return instance.rng_;
    }
  private:
    std::mt19937 rng_;
    RNG(){
      Timer ti;
      rng_.seed( (uint64_t)(1.0e6*ti.now_s()) );
    }
    RNG(const RNG&);
    void operator=(const RNG&);
};

}
#define __RANDOM_HPP
#endif /* __RANDOM_HPP */

