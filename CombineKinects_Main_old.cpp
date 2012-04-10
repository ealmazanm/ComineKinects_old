#include "CombineKinects_old.h"
#include <BackgroundSubtraction_factory.h>
#include <BackgroundDepthSubtraction.h>
#include <list>
#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/date_time/posix_time/posix_time_types.hpp"
#include <boost/date_time/posix_time/ptime.hpp>

using namespace boost::posix_time;

/*Global variable*/
const double TILT_ANGLE = 0.13962634;
const int MAX_Z = 5000; // delimits the depth of the display
const int MAX_Y = -500; // the Y axis goes from possitive to negative
ofstream outDebug(filePaths::DEBUG_FILEPATH, ios::out);
CvMat* translation_Amp;
str_actMap actMap_confVal;
bool cam1Updated = false;
bool cam2Updated = false;
int lockCameras[2];
int lockCameras_Copy[2];
//int token = 0;
bool safeImages = false;
int cont_Images = 0;

//list<str_Point> backgroundPoints;
IplImage* actMapImg = cvCreateImage(cvSize(XN_VGA_X_RES*2, XN_VGA_Y_RES), IPL_DEPTH_8U, 3);
IplImage* background_Model;
char* windName_ActiviytMap = "Activity Map";
CvMat* tilt_Rotation = cvCreateMat(3,3, CV_32FC1);
boost::mutex mymutex;

/*header methods*/
//Take the rgb image and the depth map of cam
void grabImage(CameraProperties* cam, IplImage* depthImg, XnDepthPixel* depthMap);

//Calcualte the dimensionts in 3D of the activity map. Set the size of the grids in the plane XZ
//to project the points.
void initActMapValues(CameraProperties* cam2, const XnDepthPixel* depthMap2, CameraProperties* cam1, const XnDepthPixel* depthMap1);

//Backproject the pixel (x,y) into a 3D point in the space (X,Y,Z) and check if the coordinates of the point are edges in the monitored area.
void checkCamPoint(int x, int y, const XnDepthPixel** depthMap, CameraProperties* cam, float* minX, float* maxX, float* minY, float* maxY, float* minZ, float* maxZ, bool* firstTime);

//Compare de coordinates of p3D with the minimum and maximum values in each axes. These values are updated in case any coordinate of p3D
//is a limit
void compareValues(XnPoint3D *p3D, float* minX, float* maxX, float* minY, float* maxY, float* minZ, float* maxZ);

//Calcualte the width of the grid based on the min and max values and the number of bins.
float findStep(float fmin, float fmax, int nBins);

//Transform 3D points from src to dst (Rotation and translation)
void transformPoints(vector<XnPoint3D*>* dst, vector<XnPoint3D*>* src, CameraProperties* cam);

//Generate activity map from point list 'points'
void createImage(vector<XnPoint3D*>* points, const str_actMap* actMap, int** colorMap);

//Create a combined activity map from the depthMaps
void getActivityMapImage(const str_actMap* actMap, const XnDepthPixel** depthMap1, CameraProperties* cam1 ,const XnDepthPixel** depthMap2, CameraProperties* cam2);

//Fill the activity map with the information from the depthMap. BackProject the pixels into points in the space and then project them again
// but into the ground plane. if camId = CAMID_TRANSFORMATION then the point in the space is transformed (R and t).
void fillPoints(int x, int y, const XnDepthPixel** depthMap, CameraProperties* cam, const str_actMap* actMap, IplImage* actMapImg, int** colorMap);

//Finde the projection of value in the ground plane.
int findCoordinate(float value, float minValue, float maxValue, double step);

//Transform the height into a gray scale color (0-255)
int findColor(float value, float minValue, float maxValue, double step);

//Transform the gray scale color map into a heat map color.
void findHeatColour(int alpha, int* r, int* g, int* b);

//Transform an array of 3D points from pointsIn to pointsOut (Rotation and translation)
void transformArrayPoints(XnPoint3D* points, CameraProperties& cam, int numPoints);

//Update the activity map image with the new points
void updateImage(IplImage* actMapImg, XnPoint3D* points2D, int numPoints);

//Update the background image with the points from "backgroundPoints" list
//void createBackground(IplImage* img);

/****************************************************************************************************/

/*Implementation*/
void grabImage(CameraProperties* cam, IplImage* rgbImg, XnDepthPixel* depthMap)
{
		cam->getContext()->WaitAndUpdateAll();	
		const XnDepthPixel* dM = cam->getDepthNode()->GetDepthMap();
		const XnRGB24Pixel* rgbMap = cam->getImageNode()->GetRGB24ImageMap();
		if (rgbImg != NULL)
			Utils::fillImageDataFull(rgbImg, rgbMap);

		int total = XN_VGA_X_RES*XN_VGA_Y_RES;
		for (int i = 0; i < total; i++)
			depthMap[i] = dM[i];

}

void initActMapValues(CameraProperties* cam2, const XnDepthPixel* depthMap2, CameraProperties* cam1, const XnDepthPixel* depthMap1)
{
	bool firstTime = true;
	float minX, minY, minZ;
	float maxX, maxY, maxZ;
	for (int y = 0; y < XN_VGA_Y_RES; y++)
	{
		for (int x= 0; x < XN_VGA_X_RES; x++)
		{
			//cam1
			checkCamPoint(x, y, &depthMap1, cam1, &minX, &maxX, &minY, &maxY, &minZ, &maxZ, &firstTime);
			//cam2
			checkCamPoint(x, y, &depthMap2, cam2, &minX, &maxX, &minY, &maxY, &minZ, &maxZ, &firstTime);
		}
	}
	cout << "MaxZ: " << maxZ << " MinZ: " << minZ  << endl;
	cout << "MaxY: " << maxY << " MinY: " << minY  << endl;
	cout << "MaxX: " << maxX << " MinX: " << minX  << endl;

	actMap_confVal.maxX = maxX; actMap_confVal.maxY = maxY; actMap_confVal.maxZ = MAX_Z;
	actMap_confVal.minX = minX; actMap_confVal.minY = minY; actMap_confVal.minZ = minZ;
	//Find the size of the bins (x (X), y(Z) and color(Y)). 3D-2D
	actMap_confVal.stepX = findStep(minX, maxX, XN_VGA_X_RES*2);
	actMap_confVal.stepY = findStep(minY, maxY, 256);
	actMap_confVal.stepZ = findStep(minZ, MAX_Z, XN_VGA_Y_RES);
}

float findStep(float fmin, float fmax, int nBins)
{
	float totalValues = abs(fmax - fmin);
	return totalValues/nBins;
}

void checkCamPoint(int x, int y, const XnDepthPixel** depthMap, CameraProperties* cam, float* minX, float* maxX, float* minY, float* maxY, float* minZ, float* maxZ, bool* firstTime)
{
	XnPoint3D p3D;
	int z = (*depthMap)[y*XN_VGA_X_RES+x];	
	if (z != 0)
	{
		XnPoint3D p;
		p.X = (XnFloat)x;
		p.Y = (XnFloat)y;
		p.Z = (XnFloat)z;
		cam->backProjectPoint(&p, &p3D);
		if (cam->getCamId() == CAMID_TRANSFORMATION)
			Utils::transformPoint(&p3D, *cam);
		if (*firstTime)
		{
			*minX = p3D.X; *minY = p3D.Y; *minZ = p3D.Z;
			*maxX = p3D.X; *maxY = p3D.Y; *maxZ = p3D.Z;
			*firstTime = false;
		}
		else
			compareValues(&p3D, minX, maxX, minY, maxY, minZ, maxZ); 
	}
}

void compareValues(XnPoint3D *p3D, float* minX, float* maxX, float* minY, float* maxY, float* minZ, float* maxZ)
{
	if (p3D->X < *minX)
		*minX = p3D->X;
	else if (p3D->X > *maxX)
		*maxX = p3D->X;

	if (p3D->Y < *minY)
		*minY = p3D->Y;
	else if (p3D->Y > *maxY)
		*maxY = p3D->Y;

	if (p3D->Z < *minZ)
		*minZ = p3D->Z;
	else if (p3D->Z > *maxZ)
		*maxZ = p3D->Z;				

}

void reduceTiltAngle_Array(XnPoint3D* points3D, int numPoints)
{
	CvMat* tmp = cvCreateMat(3,1, CV_32FC1);
	CvMat* tmpOut = cvCreateMat(3,1, CV_32FC1);
	for (int i = 0; i < numPoints; i++)
	{
		Utils::fillTheMatrix(tmp, &(points3D[i]), 3,1);
		cvMatMul(tilt_Rotation, tmp, tmpOut);
		points3D[i].X = (float) *(tmpOut->data.fl);
		points3D[i].Y = (float) *(tmpOut->data.fl + tmpOut->step/sizeof(float));
		points3D[i].Z = (float) *(tmpOut->data.fl + 2*tmpOut->step/sizeof(float));
	}
	
	cvReleaseMat(&tmp);
	cvReleaseMat(&tmpOut);
}

void reduceTiltAngle(vector<XnPoint3D*>* pointsCam)
{
	CvMat* tmp = cvCreateMat(3,1, CV_32FC1);
	CvMat* tmpOut = cvCreateMat(3,1, CV_32FC1);
	vector<XnPoint3D*>::iterator iter;
	for (iter = pointsCam->begin(); iter != pointsCam->end(); iter++)
	{
		XnPoint3D* p = *iter;
		Utils::fillTheMatrix(tmp, p, 3,1);
		cvMatMul(tilt_Rotation, tmp, tmpOut);
		p->X = (float) *(tmpOut->data.fl);
		p->Y = (float) *(tmpOut->data.fl + tmpOut->step/sizeof(float));
		p->Z = (float) *(tmpOut->data.fl + 2*tmpOut->step/sizeof(float));
	}

	cvReleaseMat(&tmp);
	cvReleaseMat(&tmpOut);

}


void getActivityMapImage(const str_actMap* actMap, const XnDepthPixel** depthMap1, CameraProperties* cam1 ,const XnDepthPixel** depthMap2, CameraProperties* cam2)
{
	//Color map to store the height from 0 to 255.
	int* colorMap = new int[(XN_VGA_X_RES*2)*XN_VGA_Y_RES];
	Utils::initImage3Channel(actMapImg, 0);

	vector<XnPoint3D*> pointsCam1(XN_VGA_X_RES*XN_VGA_Y_RES);
	vector<XnPoint3D*> pointsCam2(XN_VGA_X_RES*XN_VGA_Y_RES);
	vector<XnPoint3D*> pointsCam1_T(XN_VGA_X_RES*XN_VGA_Y_RES);

	//Back project all the points from both cameras
	int cont = 0;
	for (int y = 0; y < XN_VGA_Y_RES; y++)
	{
		for (int x= 0; x < XN_VGA_X_RES; x++)
		{
			XnPoint3D* p3DCam1 = new XnPoint3D;
			XnPoint3D* p3DCam2 = new XnPoint3D;
			XnPoint3D p2DCam1, p2DCam2;
			p2DCam1.X = x; p2DCam1.Y = y;
			p2DCam2.X = x; p2DCam2.Y = y;
			p2DCam1.Z = (*depthMap1)[y*XN_VGA_X_RES+x];
			p2DCam2.Z = (*depthMap2)[y*XN_VGA_X_RES+x];
		
			cam1->backProjectPoint(&p2DCam1, p3DCam1);
			cam2->backProjectPoint(&p2DCam2, p3DCam2);

			pointsCam1[cont] = p3DCam1;
			pointsCam2[cont++] = p3DCam2;
		}
	}

	//Transform all the points (R and t) from cam1-cs to cam2-cs
	transformPoints(&pointsCam1_T, &pointsCam1, cam1);

	reduceTiltAngle(&pointsCam1_T);
	reduceTiltAngle(&pointsCam2);

	createImage(&pointsCam1_T, actMap, &colorMap);
	createImage(&pointsCam2, actMap,  &colorMap);

	vector<XnPoint3D*>::iterator iterCam1, iterCam2, iterCam1T;
	iterCam1 = pointsCam1.begin();
	iterCam2 = pointsCam2.begin();
	iterCam1T = pointsCam1_T.begin();
	while (iterCam1 != pointsCam1.end())
	{
		delete(*iterCam1);
		delete(*iterCam2);
		delete(*iterCam1T);
		iterCam1++;
		iterCam2++;
		iterCam1T++;
	}

	pointsCam1.clear();
	pointsCam2.clear();
	pointsCam1_T.clear();

	delete(colorMap);
}

void updateImage(IplImage* actMapImg, XnPoint3D* points3D, int numPoints)
{
	for (int i = 0; i < numPoints; i++)
	{
		int x_2D = findCoordinate(points3D[i].X, actMap_confVal.minX, actMap_confVal.maxX, actMap_confVal.stepX);
		int y_2D = findCoordinate(points3D[i].Z, actMap_confVal.minZ, actMap_confVal.maxZ, actMap_confVal.stepZ);
	
		int y = XN_VGA_Y_RES - y_2D;
		//Create activity map using heat color for the height
		uchar* ptr_Bs = (uchar*)(actMapImg->imageData + (y*actMapImg->widthStep));
		ptr_Bs[x_2D*3] = 0;
		ptr_Bs[x_2D*3 + 1] = 0;
		ptr_Bs[x_2D*3 + 2] = 255;
	
	}
}

void createImage(vector<XnPoint3D*>* points, const str_actMap* actMap, int** colorMap)
{
	vector<XnPoint3D*>::iterator iter;
	for (iter = points->begin(); iter != points->end(); iter++)
	{
		XnPoint3D* p3D = *iter;
		if (p3D->Y > MAX_Y && p3D->Z < MAX_Z) // filter the height.
		{
			int x_2d = findCoordinate(p3D->X, actMap->minX, actMap->maxX, actMap->stepX);
			int y_2d = findCoordinate(p3D->Z, actMap->minZ, actMap->maxZ, actMap->stepZ);
			int color = findColor(p3D->Y, actMap->minY, actMap->maxY, actMap->stepY); //between 0 and 255

			//Create depth image using heat color for the height
			int r, g, b;
			findHeatColour(color, &r, &g, &b);

			//Create activity map using heat color for the height
			int y = XN_VGA_Y_RES - y_2d; //for fliping over the Y axis.
			uchar* ptr_Bs = (uchar*)(actMapImg->imageData + (y*actMapImg->widthStep));
			if ((*colorMap)[y_2d*XN_VGA_X_RES+x_2d] < color)
			{		
				//str_Point backPoint;
				//backPoint.x = x_2d;
				//backPoint.y = y_2d;
				//backPoint.r = r;
				//backPoint.g = g;
				//backPoint.b = b;
				//backgroundPoints.push_back(backPoint);

				ptr_Bs[x_2d*3] = b;
				ptr_Bs[x_2d*3 + 1] = g;
				ptr_Bs[x_2d*3 + 2] = r;
				(*colorMap)[y_2d*XN_VGA_X_RES+x_2d] = color;
			}
		}
	}
}

//void createBackground(IplImage* img)
//{
//	list<str_Point>::iterator iter;
//	for (iter = backgroundPoints.begin(); iter != backgroundPoints.end(); iter++)
//	{
//		str_Point p = *iter;
//		uchar* ptr = (uchar*)(img->imageData + (p.y*img->widthStep));
//		ptr[p.x*3] = p.b;
//		ptr[p.x*3 + 1] = p.g;
//		ptr[p.x*3 + 2] = p.r;
//	}
//}

void transformArrayPoints(XnPoint3D* points, CameraProperties& cam, int numPoints)
{
	CvMat* pointMat = cvCreateMat(3,1,CV_32FC1);
	CvMat* rotTmp = cvCreateMat(3,1,CV_32FC1);
//	CvMat* transT = cvCreateMat(3,1,CV_32FC1);
	CvMat* outMat = cvCreateMat(3,1,CV_32FC1);

//	cvTranspose(cam.getTranslationMatrix(), transT);
	XnPoint3D p;	
	for (int i = 0; i < numPoints; i++)
	{
		Utils::fillTheMatrix(pointMat, &(points[i]),3,1);
		cvMatMul(cam.getRotationMatrix(), pointMat, rotTmp);

		cvAdd(rotTmp, cam.getTranslationMatrix(), outMat);

		p.X = (XnFloat)*(outMat->data.fl);
		p.Y = (XnFloat)*(outMat->data.fl + outMat->step/sizeof(float));
		p.Z = (XnFloat)*(outMat->data.fl + 2*outMat->step/sizeof(float));

		points[i] = p;

	}
	cvReleaseMat(&pointMat);
	cvReleaseMat(&rotTmp);
	cvReleaseMat(&outMat);
//	cvReleaseMat(&transT);
}


void transformPoints(vector<XnPoint3D*>* dst, vector<XnPoint3D*>* src, CameraProperties* cam)
{
	int total  = src->size();
	//create a 3xn matrix of points
	CvMat* src_mat = cvCreateMat(3, total, CV_32FC1);
	Utils::fillTheMatrix(src, src_mat);

	//multiply by rotation and add translation
	CvMat* tmp = cvCreateMat(3, total, CV_32FC1);
	cvMatMul(cam->getRotationMatrix(), src_mat, tmp);

	CvMat* outMat = cvCreateMat(3, total, CV_32FC1);
	cvAdd(tmp, translation_Amp, outMat);


	//fill the list of points
	float *ptr_Out_C1 = (float*)outMat->data.fl;
	float *ptr_Out_C2 = (float*)outMat->data.fl + outMat->step/sizeof(float);
	float *ptr_Out_C3 = (float*)outMat->data.fl + 2*outMat->step/sizeof(float);
	for (int i = 0; i < total; i++)
	{
		XnPoint3D* p = new XnPoint3D;
		p->X = ptr_Out_C1[i];
		p->Y = ptr_Out_C2[i];
		p->Z = ptr_Out_C3[i];
		(*dst)[i] = p;
	}

	//free memory
	cvReleaseMat(&src_mat);
	cvReleaseMat(&tmp);
	cvReleaseMat(&outMat);
}

void fillPoints(int x, int y, const XnDepthPixel** depthMap, CameraProperties* cam, const str_actMap* actMap, IplImage* actMapImg, int** colorMap)
{
	XnPoint3D p3D, p;
	int z = (*depthMap)[y*XN_VGA_X_RES+x];	
	if (z != 0)
	{
		p.X = (XnFloat)x;
		p.Y = (XnFloat)y;
		p.Z = (XnFloat)z;
		cam->backProjectPoint(&p, &p3D);
		if (cam->getCamId() == CAMID_TRANSFORMATION)
			Utils::transformPoint(&p3D, *cam);
		if (p3D.Y > -500) // filter the height.
		{
			int x_2d = findCoordinate(p3D.X, actMap->minX, actMap->maxX, actMap->stepX);
			int y_2d = findCoordinate(p3D.Z, actMap->minZ, actMap->maxZ, actMap->stepZ);
			int color = findColor(p3D.Y, actMap->minY, actMap->maxY, actMap->stepY); //between 0 and 255

			//Create depth image using heat color for the height
			int r, g, b;
			findHeatColour(color, &r, &g, &b);

			//Create activity map using heat color for the height
			uchar* ptr_Bs = (uchar*)(actMapImg->imageData + (y_2d*actMapImg->widthStep));
			if ((*colorMap)[y_2d*XN_VGA_X_RES+x_2d] < color)
			{			
				ptr_Bs[x_2d*3] = b;
				ptr_Bs[x_2d*3 + 1] = g;
				ptr_Bs[x_2d*3 + 2] = r;
				(*colorMap)[y_2d*XN_VGA_X_RES+x_2d] = color;
			}
		}
	}
}

int findColor(float value, float minValue, float maxValue, double step)
{
	if (value < minValue)
		value = minValue;
	if (value > maxValue)
		value = maxValue;

	return (int)floor(-(value-maxValue)/step);

}

void findHeatColour(int alpha, int* r, int* g, int* b)
{
	int tmp;
	*r = 0;
	*g = 0;
	*b = 0;
	if(alpha <= 255 && alpha >= 215){
		tmp=255-alpha;
		*r=255-tmp;
		*g=tmp*8;
	}else if(alpha <= 214 && alpha >= 180){
		tmp=214-alpha;
		*r=255-(tmp*8);
		*g=255;
	}else if(alpha <= 179 && alpha >= 130){
		tmp=179-alpha;
		*g=255;
		*b=tmp*1;
	}else if(alpha <= 129 && alpha >= 80){
		tmp=129-alpha;
		*g=255-(tmp*1);
		*b=255;
	}else
		*b=255;	
}


int findCoordinate(float value, float minValue, float maxValue, double step)
{
	if (value < minValue)
		value = minValue;
	if (value > maxValue)
		value = maxValue;

	return (int)floor((value-minValue)/step);
}

void updateForeground(BackgroundSubtraction_factory* subtractor, CameraProperties& cam, bool transform)
{
	boost::mutex::scoped_lock mylock(mymutex, boost::defer_lock); // defer_lock makes it initially unlocked
	XnDepthPixel *depthMap = new XnDepthPixel[XN_VGA_X_RES*XN_VGA_Y_RES];
	XnPoint3D* points2D = new XnPoint3D[MAX_FORGROUND_POINTS];
	XnPoint3D* points3D = new XnPoint3D[MAX_FORGROUND_POINTS];

	char camId[20];
	_itoa(cam.getCamId(), camId, 10);
	char windName[50];
	strcpy(windName, "RGB ");
	strcat(windName, camId);
	IplImage* rgbImg = cvCreateImage(cvSize(XN_VGA_X_RES, XN_VGA_Y_RES), IPL_DEPTH_8U, 3);
	cvNamedWindow(windName);

	int numPoints = 0;
	bool stop = false;
	int cont = 0;
	while (!stop)
	{
	
	//	cvCopyImage(background_Model, actMapImg);
	
		
		grabImage(&cam, rgbImg, depthMap);

		//Save depth images.
		if (safeImages)
		{
			IplImage* depthImg = cvCreateImage(cvSize(XN_VGA_X_RES, XN_VGA_Y_RES), IPL_DEPTH_8U, 3);
				
			//create depth image
			unsigned short depth[MAX_DEPTH];
			char* depth_data = new char[640*480*3];
			Utils::raw2depth(depth, MAX_DEPTH);
			Utils::depth2rgb(depthMap, depth, depth_data);
			cvSetData(depthImg, depth_data, 640*3);
			
			char* namefile = new char[100];
			char* namefileRGB = new char[100];
			char* idCam = new char[15];
			char* idCont = new char[15];
			itoa(cam.getCamId(), idCam, 10);
			itoa(cont_Images, idCont, 10);

			strcpy(namefile, "Depth_");
			strcat(namefile, idCam);
			strcat(namefile, idCont);
			strcat(namefile, ".jpg");
			cvSaveImage(namefile, depthImg);

			strcpy(namefileRGB, "RGB_");
			strcat(namefileRGB, idCam);
			strcat(namefileRGB, idCont);
			strcat(namefileRGB, ".jpg");
			cvSaveImage(namefileRGB, rgbImg);

			cvReleaseImageHeader(&depthImg);
			delete(depth_data);
			delete(namefile);
			delete(idCont);
			delete(idCam);
			
		}


		numPoints = subtractor->subtraction(points2D, depthMap); //returns the num poins of foreground

		//Save binary image with the background subtraction
		if (safeImages)
		{
			IplImage* backImg = cvCreateImage(cvSize(XN_VGA_X_RES, XN_VGA_Y_RES), IPL_DEPTH_8U, 1);
			Utils::initImage(backImg, 0);
			for (int i = 0; i < numPoints; i++)
			{
				XnPoint3D p = points2D[i];
				uchar *ptrBackImg = (uchar*)backImg->imageData + (int)p.Y*backImg->widthStep;
				ptrBackImg[(int)p.X] = 255;
			}
			char* namefile = new char[100];
			char* idCam = new char[15];
			char* idCont = new char[15];
			itoa(cam.getCamId(), idCam, 10);
			itoa(cont_Images, idCont, 10);

			strcpy(namefile, "BackGroundSub_");
			strcat(namefile, idCam);
			strcat(namefile, idCont);
			strcat(namefile, ".jpg");
			cvSaveImage(namefile, backImg);
					
			delete(namefile);
			delete(idCam);
			cvReleaseImage(&backImg);

		}


		Utils::backProjectArrayOfPoints(points3D, points2D, cam, numPoints);
		if (transform)
			transformArrayPoints(points3D, cam, numPoints);

		////update actMapImg with the points
		
//		reduceTiltAngle_Array(points3D, numPoints);
		mylock.lock();
		{
			if (cam.getCamId() == 1)
			{
				if (!cam2Updated)
					cvCopyImage(background_Model, actMapImg);
			}
			else
			{
				if (!cam1Updated)
					cvCopyImage(background_Model, actMapImg);
			}
			updateImage(actMapImg, points3D, numPoints);
			if (cam.getCamId() == 1)
			{
				
				if (cam2Updated)
				{
					cvShowImage(windName_ActiviytMap, actMapImg);

					if (safeImages)
					{
						char* namefile = new char[100];
						char* idCont = new char[15];
						itoa(cont_Images, idCont, 10);

						strcpy(namefile, "ActivityMap_");
						strcat(namefile, idCont);
						strcat(namefile, ".jpg");
						cvSaveImage(namefile, actMapImg);
						delete(namefile);
						safeImages = false;
						cont_Images++;
					}

					cam2Updated = false;
					
				}
				else
					cam1Updated = true;
			}
			else
			{
				if (cam1Updated)
				{
					cvShowImage(windName_ActiviytMap, actMapImg);

					if (safeImages)
					{
						char* namefile = new char[100];
						char* idCont = new char[15];
						itoa(cont_Images, idCont, 10);

						strcpy(namefile, "ActivityMap_");
						strcat(namefile, idCont);
						strcat(namefile, ".jpg");
						cvSaveImage(namefile, actMapImg);
						delete(namefile);
						safeImages = false;
						cont_Images++;
					}


					cam1Updated = false;
					
				}
				else
					cam2Updated = true;
			}

		}
		mylock.unlock();
//		cvShowImage(windName_ActiviytMap, actMapImg);
		//if (cam.getCamId() == 1)
		//{
		//	lockCameras[0] = 1;
		//	if (lockCameras[1] == 1)
		//		cvShowImage(windName_ActiviytMap, actMapImg);
		//	else
		//		while(lockCameras[1] == 0);
		//}
		//else
		//{
		//	lockCameras[1] = 1;
		//	if (lockCameras[0] == 1)
		//		cvShowImage(windName_ActiviytMap, actMapImg);
		//	else
		//		while(lockCameras[0] == 0);
		//}
		
		//Only the last threads display the image
//		while (token != 0)
//		{
//			outDebug << "cam " << cam.getCamId() << " waiting. Token: " << token << endl;
//			boost::this_thread::sleep(boost::posix_time::seconds(0.1));
//		}
		
			cvShowImage(windName, rgbImg);
			char c = cvWaitKey(1);
			stop = (c == 27) || (cont > 350);
			safeImages = (c==115);
//		}
		//cont++;
	}

	delete(points2D);
	delete(points3D);
	delete(depthMap);
	cvReleaseImage(&rgbImg);
}

void displayActivityMap()
{
//	boost::mutex::scoped_lock mylock(mymutex, boost::defer_lock); // defer_lock makes it initially unlocked
	bool stop = false;
	while (!stop)
	{
		if ((lockCameras[0] == 1) && (lockCameras[1] == 1))
		{
			cvShowImage(windName_ActiviytMap, actMapImg);
			char c = cvWaitKey(1);
			stop = (c == 27);
			lockCameras[0] == 0;
			lockCameras[0] == 0;
		}
	}
	
}

int main()
{
	//Init all sensors (init context)
	CameraProperties cam1, cam2;
	Utils::rgbdInit(&cam1, &cam2);
	Utils::loadCameraParameters(&cam1);
	Utils::loadCameraParameters(&cam2);

	lockCameras[0] = 0;
	lockCameras[1] = 0;
	lockCameras_Copy[0] = 0;
	lockCameras_Copy[1] = 0;

	//Tilt rotation matrix
	CvMat* rot = cvCreateMat(3,1, CV_32FC1);
	float* rotPtr_1 = (float*)(rot->data.fl);
	float* rotPtr_2 = (float*)(rot->data.fl + rot->step/sizeof(float));
	float* rotPtr_3 = (float*)(rot->data.fl + (2*rot->step/sizeof(float)));
	*rotPtr_1 = TILT_ANGLE; // 5 degrees over the x axis
	*rotPtr_2 = 0;
	*rotPtr_3 = 0;
	cvRodrigues2(rot, tilt_Rotation); 

	//Init tranlsation matrix
	int total = XN_VGA_X_RES*XN_VGA_Y_RES;
	translation_Amp = cvCreateMat(3, total, CV_32FC1);
	for (int r = 0; r < 3; r++)
	{
		float val_T = (float)cam1.getTranslationMatrix()->data.fl[r];
		float *ptr_Amp = (float*)translation_Amp->data.fl + (r*translation_Amp->step/(sizeof(float)));
		for (int i = 0 ; i  < total; i++)
		{
			ptr_Amp[i] = val_T;
		}
	}


	//declare image variables
	IplImage* rgbImg1 = cvCreateImage(cvSize(XN_VGA_X_RES, XN_VGA_Y_RES), IPL_DEPTH_8U, 3);
	IplImage* rgbImg2 = cvCreateImage(cvSize(XN_VGA_X_RES, XN_VGA_Y_RES), IPL_DEPTH_8U, 3);
	XnDepthPixel *depthMap1, *depthMap2;

	depthMap1 = new XnDepthPixel[XN_VGA_X_RES*XN_VGA_Y_RES];
	depthMap2 = new XnDepthPixel[XN_VGA_X_RES*XN_VGA_Y_RES];

	//Start capturing images from cameras 1 and 2.
	cam1.getContext()->StartGeneratingAll();
	cam2.getContext()->StartGeneratingAll();

	//take images (threads) 
	boost::thread thr(grabImage, &cam2, rgbImg1, depthMap2);
	grabImage(&cam1, rgbImg2, depthMap1);
	thr.join();

	cvReleaseImage(&rgbImg1);
	cvReleaseImage(&rgbImg2);

	initActMapValues(&cam2, depthMap2, &cam1, depthMap1);
	BackgroundSubtraction_factory* subtractor1 = new BackgroundDepthSubtraction(depthMap1);
	BackgroundSubtraction_factory* subtractor2 = new BackgroundDepthSubtraction(depthMap2);
	
	cvNamedWindow(windName_ActiviytMap);
		
	//create activity map for both sensors
	getActivityMapImage(&actMap_confVal, (const XnDepthPixel**)&depthMap1, &cam1,(const XnDepthPixel**)&depthMap2, &cam2);
	//	cvFlip(actMapImg, NULL,0);
	background_Model = cvCloneImage(actMapImg);


//	ptime time_start_wait(microsec_clock::local_time());
//	boost::thread thr1(displayActivityMap);
	boost::thread thr2(updateForeground, subtractor2, cam2, false);
	updateForeground(subtractor1, cam1, true);
	thr2.join();
//	thr1.join();

	//Take time
//	ptime time_end_wait(microsec_clock::local_time());
//	time_duration duration_wait(time_end_wait - time_start_wait);
//	double frames = (350/duration_wait.total_seconds());
//	cout << "fps: " << frames << endl;
	
	//Teminate capturing images
	cam1.getContext()->StopGeneratingAll();
	cam2.getContext()->StopGeneratingAll();
	//Free memory
	delete(depthMap1);
	delete(depthMap2);
	cvDestroyAllWindows();
	cvReleaseImage(&actMapImg);
	cvReleaseMat(&translation_Amp);
	return 0;
}