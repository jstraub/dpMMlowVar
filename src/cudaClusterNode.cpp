#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>

/* Cuda Cluster */
#include <dpMMlowVar/ddpmeansCUDA.hpp>
#include <jsCore/clDataGpu.hpp>
#include <dpMMlowVar/euclideanData.hpp>


class CudaClusterNode{
public:
    ros::NodeHandle nh_;
    ros::Subscriber sub_cloud_;

    /* Clustering */
    // dplv::Clusterer<double, dplv::Euclidean<double> > *clusterer_;
    dplv::DDPMeansCUDA<double, dplv::Euclidean<double> > *clusterer_;
    shared_ptr<jsc::ClDataGpu<double> > cld_;
    shared_ptr<Eigen::MatrixXd> spx_;

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
        double Q = lambda_/t_q_;
        double tau = (t_q_*(k_tau_-1.0)+1.0)/(t_q_-1.0);

        /* Initialize the clusterer */
        // clusterer_ = NULL;

        // shared_ptr<Eigen::MatrixXd> spx(new MatrixXd(2,1));
        spx_ = shared_ptr<Eigen::MatrixXd>(new MatrixXd(2,1));
        Eigen::MatrixXd& input_matrix(*spx_);
        input_matrix(0,0) = 0.0;
        input_matrix(1,0) = 0.0;
        cld_ = shared_ptr<jsc::ClDataGpu<double> >(new jsc::ClDataGpu<double>(spx_,0));
        clusterer_ = new dplv::DDPMeansCUDA<double, dplv::Euclidean<double> >(cld_, lambda_, Q, tau);

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
        ros::param::getCached("lambda",lambda_);
        ros::param::getCached("t_q",t_q_);
        ros::param::getCached("k_tau",k_tau_);
        ros::param::getCached("max_kmean_iter",max_kmean_iter_);
    }

    void cb_cloud(const sensor_msgs::PointCloud2& input_cloud)
    {
        ROS_INFO_STREAM("[CudaClusterNode] Callback.");
        ros::Time start_time = ros::Time::now();

        /* Convert to pcl::PointCloud */
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg (input_cloud, *cloud);

        /* Populate the matrix for clusterer */
        // shared_ptr<Eigen::MatrixXd> spx(new MatrixXd(2,cloud->size()));
        spx_->resize(2,cloud->size());
        Eigen::MatrixXd& input_matrix(*spx_);
        for(pcl::PointCloud<pcl::PointXYZ>::const_iterator iter = cloud->begin(); iter != cloud->end(); ++iter)
        {
            int N_index = iter - cloud->begin();
            input_matrix(0,N_index) = (*iter).x;
            input_matrix(1,N_index) = (*iter).y;
        }
        ROS_INFO_STREAM("[CudaClusterNode] Matrix populated. rows: " << spx_->rows() << " cols: " << spx_->cols() );

        clusterer_->nextTimeStep(spx_);
        ROS_INFO_STREAM("[CudaClusterNode] clusterer feed.");

        /* Do clustering and gather results */
        Eigen::MatrixXd deviates;
        Eigen::MatrixXd centroids;
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