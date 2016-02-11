#include "Vocabulary.h"


Vocabulary::Vocabulary()
{
    //ctor
}

Vocabulary::~Vocabulary()
{
    //dtor
}

void Vocabulary::clear()
{
    indexedDescriptors_ = cv::Mat();
    notIndexedDescriptors_ = cv::Mat();
    wordToObjects_.clear();
    notIndexedWordIds_.clear();

}

cv::flann::SearchParams Vocabulary::getFlannSearchParams()
{
    int checks=32;
    float eps=0;
    bool sorted=true;
	return cv::flann::SearchParams(
			32, 0, true);
}

cvflann::flann_distance_t Vocabulary::getFlannDistanceType()
{
	cvflann::flann_distance_t distance = cvflann::FLANN_DIST_L2;

	return distance;
}

cv::flann::IndexParams * Vocabulary::createFlannIndexParams(int index)
{
	cv::flann::IndexParams * params = 0;

    switch(index)
    {
        case 0:
            params = new cv::flann::LinearIndexParams();
            cout<<"flann linearIndexParams"<<endl;
            break;
        case 1:
            params = new cv::flann::KDTreeIndexParams();
            cout<<"flann randomized kdtrees params"<<endl;
            break;
        case 2:
            params = new cv::flann::KMeansIndexParams();
            cout<<"flann hierarchical k-means tree params"<<endl;
            break;
        case 3:
            params = new cv::flann::CompositeIndexParams();
            cout<<"flann combination of randomized kd-trees and hierarchical k-means tree"<<endl;
            break;
        /*case 4:
            params = new cv::flann::LshIndexParams();
            cout<<"flann LSH Index params"<<endl;
            break;*/
        case 5:
            params = new cv::flann::AutotunedIndexParams();
            cout<<"flann AutotunedIndexParams"<<endl;
            break;
        default:
            break;
    }

	if(!params)
	{
		printf("ERROR: NN strategy not found !? Using default KDTRee...\n");
		params = new cv::flann::KDTreeIndexParams();
	}
	return params ;
}

void Vocabulary::update()
{
	if(!notIndexedDescriptors_.empty())
	{
		assert(indexedDescriptors_.cols == notIndexedDescriptors_.cols &&
				 indexedDescriptors_.type() == notIndexedDescriptors_.type() );

		//concatenate descriptors
		indexedDescriptors_.push_back(notIndexedDescriptors_);  //插入新进来的图片的IndexedDescriptors

		notIndexedDescriptors_ = cv::Mat();
		notIndexedWordIds_.clear(); //把新加进来的notIndexedwords清空
	}

	if(!indexedDescriptors_.empty())
	{
		cv::flann::IndexParams * params = createFlannIndexParams(1);
		flannIndex_.build(indexedDescriptors_, *params, getFlannDistanceType());
		delete params;
	}
}

void Vocabulary::search(const cv::Mat & descriptors, cv::Mat & results, cv::Mat & dists, int k)
{
	assert(notIndexedDescriptors_.empty() && notIndexedWordIds_.size() == 0);

	if(!indexedDescriptors_.empty())
	{
		assert(descriptors.type() == indexedDescriptors_.type() && descriptors.cols == indexedDescriptors_.cols);

		flannIndex_.knnSearch(descriptors, results, dists, k, getFlannSearchParams());

		if( dists.type() == CV_32S )
		{
			cv::Mat temp;
			dists.convertTo(temp, CV_32F);
			dists = temp;
		}
	}
}
