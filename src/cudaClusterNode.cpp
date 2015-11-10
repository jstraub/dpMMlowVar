#include <ros/ros.h>
// #include <algorithm>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
// #include <utilpcl.hpp>

/* Cuda Cluster */
#include <dpMMlowVar/kmeans.hpp>
#include <dpMMlowVar/dpmeans.hpp>
#include <dpMMlowVar/ddpmeans.hpp>

// using namespace Eigen;
// using namespace std;
// using namespace dplv;

class CudaClusterNode{
public:
    ros::NodeHandle nh_;
    ros::Subscriber sub_cloud_;

    /* Clustering */
    dplv::Clusterer<double, dplv::Euclidean<double> > *clusterer_;
    /* Parameters */
    double lambda_;
    double t_q_;
    double k_tau_;
    bool revive_on_init_;
    int max_kmean_iter_; //Maximum number of kmean iteration?

    CudaClusterNode()
    {
        nh_ = ros::NodeHandle("~");
        setDefaultParams();
        getParams();
        clusterer_ = NULL;
        /* Subscription */
        sub_cloud_ = nh_.subscribe("input_cloud",1,&CudaClusterNode::cb_cloud,this);
    }

    ~CudaClusterNode(){}

    void setDefaultParams()
    {
        if (!ros::param::has("~lambda")) ros::param::set("~lambda",1.0);
        if (!ros::param::has("~t_q")) ros::param::set("~t_q",30);
        if (!ros::param::has("~k_tau")) ros::param::set("~k_tau",1.2);
        if (!ros::param::has("~revive_on_init")) ros::param::set("~revive_on_init",true);
        if (!ros::param::has("~max_kmean_iter")) ros::param::set("~max_kmean_iter",1000);
    }

    void getParams()
    {
        ros::param::getCached("lambda",lambda_);
        ros::param::getCached("t_q",t_q_);
        ros::param::getCached("k_tau",k_tau_);
        ros::param::getCached("revive_on_init",revive_on_init_);
        ros::param::getCached("max_kmean_iter",max_kmean_iter_);
    }

    void cb_cloud(const sensor_msgs::PointCloud2& input_cloud)
    {
        ros::Time start_time = ros::Time::now();
        /* Convert to pcl::PointCloud */
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg (input_cloud, *cloud);

        /* Get number of points */
        size_t N = cloud->size();
        shared_ptr<Eigen::MatrixXd> spx(new MatrixXd(2,N));
        Eigen::MatrixXd& input_matrix(*spx);

        /* Populate the matrix for clusterer */
        for(pcl::PointCloud<pcl::PointXYZ>::const_iterator iter = cloud->begin(); iter != cloud->end(); ++iter)
        {
            int N_index = iter - cloud->begin();
            input_matrix(0,N_index) = (*iter).x;
            input_matrix(1,N_index) = (*iter).y;
        }

        if (clusterer_ != NULL){
            /*Initialize clusterer */
            clusterer_ = new dplv::DDPMeans<double, dplv::Euclidean<double> >(spx, lambda_, t_q_, k_tau_);
        }
        else{
            /*Feed the clusterer */
            clusterer_->nextTimeStep(spx);
        }

        /* Do clustering and gather results */
        Eigen::MatrixXd deviates;
        Eigen::MatrixXd centroids;
        MatrixXu inds; //typedef in jsCore/global.hpp
        for (size_t i = 0; i < max_kmean_iter_; ++i){
            clusterer_->updateCenters();
            clusterer_->updateLabels();
            if(clusterer_->converged()) break;
        }
        clusterer_->updateState();
        // silhouette = clustEu->silhouette();
        centroids = clusterer_->centroids();
        inds = clusterer_->mostLikelyInds(10,deviates);

        ros::Time end_time = ros::Time::now();
        /* Print out */
        ROS_INFO_STREAM("[CudaClusterNode] Number of clusters: " << centroids.cols());
        ROS_INFO("[CudaClusterNode] Took %.10f", (end_time - start_time).toSec());
    }
};

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "cuda_cluster_node");
    CudaClusterNode cuda_cluster_node;
    ros::spin();
    return 0;
}