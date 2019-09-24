// All includes
#include <sl/Camera.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <iostream>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/common/common.h>
#include <boost/thread/thread.hpp>
#include <thread>
#include <mutex>

// Undef on Win32 min/max for PCL
#ifdef _WIN32
#undef max
#undef min
#endif

// Namespace
using namespace sl;
using namespace std;

// Global instance (ZED, Mat, callback)
Camera zed;
Mat data_cloud;
std::thread zed_callback;
std::mutex mutex_input;
bool stop_signal;
bool has_data;

// Sample functions
void startZED();
void run();
void closeZED();

inline float convertColor(float colorIn);

// Main process
int main(int argc, char **argv) {

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>), cloud_f(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud->points.resize(zed.getResolution().area());

    // Create the PCL point cloud visualizer
    shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("PCL ZED 3D Viewer"));
    viewer->setBackgroundColor(0.12, 0.12, 0.12);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    shared_ptr<pcl::visualization::PCLVisualizer> viewer_cluster(new pcl::visualization::PCLVisualizer("Extracted Clusters"));
    viewer_cluster->setBackgroundColor(0.12, 0.12, 0.12);
    viewer_cluster->addCoordinateSystem(1.0);
    viewer_cluster->initCameraParameters();
    viewer_cluster->setRepresentationToWireframeForAllActors();

    // Set configuration parameters
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION_HD720;
//    if (argc == 2)
//        init_params.svo_input_filename = argv[1];
    init_params.coordinate_units = UNIT_METER;
    init_params.coordinate_system = COORDINATE_SYSTEM_LEFT_HANDED_Y_UP;
    init_params.depth_mode = DEPTH_MODE_PERFORMANCE;

    // Open the camera
    ERROR_CODE err = zed.open(init_params);
    if (err != SUCCESS) {
        cout << errorCode2str(err) << endl;
        zed.close();
        return 1;
    }

    // Allocate PCL point cloud at the resolution
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr p_pcl_point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    p_pcl_point_cloud->points.resize(zed.getResolution().area());

    // Start ZED callback
    startZED();

    // Loop infinitely
    while (!viewer->wasStopped()) {

        // Try to lock the data if possible (not in use). Otherwise, do nothing.
        if (mutex_input.try_lock()) {
            float *p_data_cloud = data_cloud.getPtr<float>();
            int index = 0;

            // Check and adjust points for PCL format
            for (auto &it : p_pcl_point_cloud->points) {
                float X = p_data_cloud[index];
                if (!isValidMeasure(X)) // Checking if it's a valid point
                    it.x = it.y = it.z = 0;
                else {
                    it.x = X;
                    it.y = p_data_cloud[index + 1];
                    it.z = p_data_cloud[index + 2];
                    it.rgb = convertColor(p_data_cloud[index + 3]); //Convert a 32bits float into a pcl rgb format
                }
                index += 4;
            }

            // Unlock data and update Point cloud
            mutex_input.unlock();

            // Create the filtering object: downsample the dataset using a leaf size of 1cm
            pcl::VoxelGrid<pcl::PointXYZRGB> vg;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
            vg.setInputCloud(p_pcl_point_cloud);
            vg.setLeafSize(0.1f, 0.1f, 0.1f); // leaf sife
            vg.filter(*cloud_filtered);
            //std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size() << " data points." << std::endl;

            // Create the segmentation object for the planar model and set all the parameters
            pcl::SACSegmentation<pcl::PointXYZRGB> seg;
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZRGB>());
            pcl::PCDWriter writer;
            //Set up parameters for our segmentation/ extraction scheme
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE ); //only want points perpendicular to a given axis
            seg.setMaxIterations(500); // this is key (default is 50 and that sucks)
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold (0.05); // keep points within 0.05 m of the plane

            //because we want a specific plane (X-Y Plane) (In camera coordinates the ground plane is perpendicular to the y axis)
            Eigen::Vector3f axis = Eigen::Vector3f(0.0,0.0,1.0); //Z axis
            seg.setAxis(axis);
            seg.setEpsAngle(  30.0f * (M_PI/180.0f) ); // plane can be within 30 degrees of X-Y plane

            int i = 0, nr_points = (int)cloud_filtered->points.size();
            while (cloud_filtered->points.size() > 0.3 * nr_points)
            {
                // Segment the largest planar component from the remaining cloud
                seg.setInputCloud(cloud_filtered);
                seg.segment(*inliers, *coefficients);
                if (inliers->indices.size() == 0)
                {
                    std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
                    break;
                }

                // Extract the planar inliers from the input cloud
                pcl::ExtractIndices<pcl::PointXYZRGB> extract;
                extract.setInputCloud(cloud_filtered);
                extract.setIndices(inliers);
                extract.setNegative(false);

                // Get the points associated with the planar surface
                extract.filter(*cloud_plane);
                //std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size() << " data points." << std::endl;

                // Remove the planar inliers, extract the rest
                extract.setNegative(true);
                extract.filter(*cloud_f);
                *cloud_filtered = *cloud_f;
            }

            // Creating the KdTree object for the search method of the extraction
            pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
            tree->setInputCloud(cloud_filtered);

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
            ec.setClusterTolerance(0.1);//设置近邻搜索的搜索半径为10cm
            ec.setMinClusterSize(50);//设置一个聚类需要的最少点数目为100
            ec.setMaxClusterSize(500);//设置一个聚类需要的最大点数目为500
            ec.setSearchMethod(tree);//设置点云的搜索机制
            ec.setInputCloud(cloud_filtered);
            ec.extract(cluster_indices);

            //clear clouds
            viewer_cluster->removeAllShapes();
            viewer_cluster->removeAllPointClouds();
            viewer_cluster->setRepresentationToWireframeForAllActors();

            int j = 0;
            for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
            {
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
                for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
                    cloud_cluster->points.push_back(cloud_filtered->points[*pit]);
                cloud_cluster->width = cloud_cluster->points.size();
                cloud_cluster->height = 1;
                cloud_cluster->is_dense = true;
                pcl::PointXYZRGB minPt, maxPt;

                pcl::getMinMax3D(*cloud_cluster, minPt, maxPt);
                float distance = sqrt((minPt.x+maxPt.x) * (minPt.x+maxPt.x)/4 + (minPt.y+maxPt.y) * (minPt.y+maxPt.y)/4 + (minPt.z+maxPt.z) * (minPt.z+maxPt.z)/4);
                std::cout << "Max x: " << maxPt.x << " m " << std::endl;
                std::cout << "Max y: " << maxPt.y << " m " << std::endl;
                std::cout << "Max z: " << maxPt.z << " m " << std::endl;
                std::cout << "Min x: " << minPt.x << " m " << std::endl;
                std::cout << "Min y: " << minPt.y << " m " << std::endl;
                std::cout << "Min z: " << minPt.z << " m " << std::endl;
                std::cout << "------------------------------------------------------------------------------------"<< std::endl;
                std::cout << "Center of cluster cloud ( " << (minPt.x+maxPt.x)/2 << ", " << (minPt.y+maxPt.y)/2 << ", "<< (minPt.z+maxPt.z)/2 << " )    Distance is " << distance << " m " << std::endl;
                std::cout << "------------------------------------------------------------------------------------"<< std::endl;
                float x_min = minPt.x;
                float x_max = maxPt.x;
                float y_min = minPt.y;
                float y_max = maxPt.y;
                float z_min = minPt.z;
                float z_max = maxPt.z;
                viewer_cluster->setRepresentationToWireframeForAllActors();
                viewer_cluster->addCube(x_min, x_max, y_min, y_max, z_min, z_max, 1.0, 0, 0, std::to_string(j+50), 0);
                viewer_cluster->setRepresentationToWireframeForAllActors();
                viewer_cluster->addPointCloud<pcl::PointXYZRGB>(cloud_cluster, std::to_string(j));
                viewer_cluster->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, std::to_string(j));
                viewer_cluster->setRepresentationToWireframeForAllActors();
                //std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points." << std::endl;
                j++;
            }
            viewer->updatePointCloud(p_pcl_point_cloud, "cloud");
            viewer->spinOnce(1000);
            viewer_cluster->spinOnce(1000);
            viewer_cluster->setRepresentationToWireframeForAllActors();
        }
        else
            sleep_ms(1);
    }

    // Close the viewer
    viewer->close();

    // Close the zed
    closeZED();

    return 0;
}

/**
 * This functions start the ZED's thread that grab images and data.
 */
void startZED() {
    // Start the thread for grabbing ZED data
    stop_signal = false;
    has_data = false;
    zed_callback = std::thread(run);

    //Wait for data to be grabbed
    while (!has_data)
        sleep_ms(1);
}

/**
 * This function loops to get the point cloud from the ZED. It can be considered as a callback.
 */
void run() {
    while (!stop_signal)
    {
        if (zed.grab(SENSING_MODE_STANDARD) == SUCCESS)
        {
            mutex_input.lock(); // To prevent from data corruption
            zed.retrieveMeasure(data_cloud, MEASURE_XYZRGBA);
            mutex_input.unlock();
            has_data = true;
        }
        sleep_ms(1);
    }
}

/**
 * This function frees and close the ZED, its callback(thread) and the viewer
 */
void closeZED() {
    // Stop the thread
    stop_signal = true;
    zed_callback.join();
    zed.close();
}

/**
 *  This function convert a RGBA color packed into a packed RGBA PCL compatible format
 **/
inline float convertColor(float colorIn) {
    uint32_t color_uint = *(uint32_t *) & colorIn;
    unsigned char *color_uchar = (unsigned char *) &color_uint;
    color_uint = ((uint32_t) color_uchar[0] << 16 | (uint32_t) color_uchar[1] << 8 | (uint32_t) color_uchar[2]);
    return *reinterpret_cast<float *> (&color_uint);
}
