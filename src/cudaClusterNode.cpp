#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>

/* Cuda Cluster */
#include <dpMMlowVar/ddpmeansCUDA.hpp>
#include <jsCore/clDataGpu.hpp>
#include <dpMMlowVar/euclideanData.hpp>


typedef shared_ptr<dplv::DDPMeansCUDA<float, dplv::Euclidean<float> > > DDPMeansCUDAPtr;

class CudaClusterNode{
public:
    ros::NodeHandle nh_;
    ros::Subscriber sub_cloud_;

    /* Clustering */
    DDPMeansCUDAPtr clusterer_;
    shared_ptr<jsc::ClDataGpu<float> > cld_;
    shared_ptr<Eigen::MatrixXf> spx_;

    /* Parameters */
    double lambda_;
    double t_q_;
    double k_tau_;
    // bool revive_on_init_;
    int max_kmean_iter_; //Maximum number of kmean iteration?

    CudaClusterNode()
    {
        nh_ = ros::NodeHandle("~");
        setDefaultParams();
        getParams();
        /* Initialize spx_ */
        shared_ptr<Eigen::MatrixXf> spx(new MatrixXf(3,0));
        spx_ = spx;

        /* Subscription */
        sub_cloud_ = nh_.subscribe("input_cloud",1,&CudaClusterNode::cb_cloud,this);
    }

    ~CudaClusterNode(){}

    void setDefaultParams()
    {
        if (!ros::param::has("~lambda")) ros::param::set("~lambda",1.0);
        if (!ros::param::has("~t_q")) ros::param::set("~t_q",30);
        if (!ros::param::has("~k_tau")) ros::param::set("~k_tau",1.2);
        if (!ros::param::has("~max_kmean_iter")) ros::param::set("~max_kmean_iter",1000);
    }

    void getParams()
    {
        ros::param::getCached("~lambda",lambda_);
        ros::param::getCached("~t_q",t_q_);
        ros::param::getCached("~k_tau",k_tau_);
        ros::param::getCached("~max_kmean_iter",max_kmean_iter_);
    }

    void cb_cloud(const sensor_msgs::PointCloud2& input_cloud)
    {
        ROS_INFO_STREAM("[CudaClusterNode] Callback.");
        ros::Time start_time = ros::Time::now();

        /* Convert to pcl::PointCloud */
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg (input_cloud, *cloud);
        /* Populate the matrix for clusterer */
        spx_->resize(3,cloud->size());
        Eigen::MatrixXf& input_matrix(*spx_);
        for(pcl::PointCloud<pcl::PointXYZ>::const_iterator iter = cloud->begin(); iter != cloud->end(); ++iter)
        {
            int N_index = iter - cloud->begin();
            input_matrix(0,N_index) = (*iter).x;
            input_matrix(1,N_index) = (*iter).y;
            input_matrix(2,N_index) = 0.0;
        }
        ROS_INFO_STREAM("[CudaClusterNode] Matrix populated. rows: " << spx_->rows() << " cols: " << spx_->cols() );
        // ROS_INFO_STREAM("[CudaClusterNode] Matrix: \n" << input_matrix.transpose());

        if (clusterer_ == NULL){
            /* Initialize the clusterer_ at the first callback */
            float Q = lambda_/t_q_;
            float tau = (t_q_*(k_tau_-1.0)+1.0)/(t_q_-1.0);
            cld_ = shared_ptr<jsc::ClDataGpu<float> >(new jsc::ClDataGpu<float>(spx_,0));
            ROS_INFO_STREAM("[CudaClusterNode] cld_ initialized. lambda=" << lambda_ << " Q = " << Q << " tau = " << tau);
            clusterer_ = DDPMeansCUDAPtr(new dplv::DDPMeansCUDA<float, dplv::Euclidean<float> >(cld_, lambda_, Q, tau));
            ROS_INFO_STREAM("[CudaClusterNode] Clusterer Initialized.");
        }
        else{
            /* Call nextTimeStep if initialized already */
            // clusterer_->nextTimeStep(spx_,false);
            clusterer_->nextTimeStep(spx_,true);
            ROS_INFO_STREAM("[CudaClusterNode] Clusterer nextTimeStep called.");
        }

        /* Do clustering and gather results */
        Eigen::MatrixXf deviates;
        Eigen::MatrixXf centroids;
        MatrixXu inds; //typedef in jsCore/global.hpp
        for (size_t i = 0; i < max_kmean_iter_; ++i){
            ROS_INFO_STREAM("[CudaClusterNode] kmean iter: " << i);
            clusterer_->updateLabels();
            ROS_INFO_STREAM("[CudaClusterNode] updateLabels() done.");
            clusterer_->updateCenters();
            ROS_INFO_STREAM("[CudaClusterNode] updateCenters() done.");
            if(clusterer_->converged()) break;
        }


        ROS_INFO_STREAM("[CudaClusterNode] Done kmean iterations.");
        clusterer_->updateState();
        ROS_INFO_STREAM("[CudaClusterNode] Done updateState.");
        // silhouette = clustEu->silhouette();
        centroids = clusterer_->centroids();
        inds = clusterer_->mostLikelyInds(10,deviates);
        ROS_INFO_STREAM("[CudaClusterNode] Number of clusters: " << centroids.cols());

        ros::Time end_time = ros::Time::now();
        /* Print out */
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