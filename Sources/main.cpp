//Require OPENCV and Tensorflow
//This program could be much faster



#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <time.h>

#include "Rect_fuser.h"

#define IMG_X 896 
#define IMG_Y 592 
#define IMG_Z 3 

#define SIZE_SQUARE 228
#define STRIDE_X (SIZE_SQUARE/4.0 )
#define STRIDE_Y (SIZE_SQUARE/4.0 )
#define BATCH_SIZE 10
#define NUMBER_OF_ITER 50000
#define NUMBER_OF_IMAGES 400
#define READ_DATABASE 0
#define SAVE_RUN_BB 0
#define SAVE_RUN_RECT 0
#define SAVE_RUN_SIZE 0


using namespace std;
using namespace cv;
using namespace tensorflow;

class database_unit_training
{
	public :
	float image_12[12*12*3];
	int is_face;
	float dx;
	float dy;
	float dw;
	float dh;
	Rect original;
	float confidence;
};

int overlap_ratio(Rect aabb1, Rect aabb2)
{
	double XA1 = (aabb1.x);
	double XA2 = (aabb1.x+aabb1.width);
	double XB1 = (aabb2.x);
	double XB2 = (aabb2.x+aabb2.width);
	double YA1 = (aabb1.y);
	double YA2 = (aabb1.y+aabb1.height);
	double YB1 = (aabb2.y);
	double YB2 = (aabb2.y+aabb2.height);
	double SA = (aabb1.width)*(aabb1.height);
	double SB = (aabb2.width)*(aabb2.height);
	double SSQUARE = aabb2.width*aabb2.width;
	double SI = max(0.0, min(XA2, XB2) - max(XA1, XB1)) * max(0.0, min(YA2, YB2) - max(YA1, YB1));
	double S = SA+SB-SI;
	if(SI/S > 0.3 || SI/SSQUARE > 0.9) return 1;
	return 0;
}

Point get_translation_vis12rect_vign(float image_12[12*12*3], Session* VIS_12_RECT)
{
	Tensor network_inputs(DT_FLOAT, TensorShape({BATCH_SIZE, 12* 12* 3}));
	Tensor network_outputs(DT_FLOAT, TensorShape({BATCH_SIZE, 2}));
	std::vector<Tensor> outputs;
	auto _InputTensor = network_inputs.tensor<float, 2>();
	auto _OutputTensor = network_outputs.tensor<float, 2>();

	memcpy(&_InputTensor(0, 0), image_12, 12*12*3*sizeof(float));
	TF_CHECK_OK(VIS_12_RECT->Run({{"x", network_inputs}, {"y", network_outputs}}, {"y_out"}, {}, &outputs));
	auto true_output = outputs[0].tensor<float, 2>();

	Point destination(true_output(0,0)*SIZE_SQUARE, true_output(0,1)*SIZE_SQUARE);
	return destination;
}

database_unit_training create_unit_12x12(Mat* resized_frame){
	database_unit_training returned_unit;
	int indice = 0;
	for(int vign_i=0 ; vign_i<12 ; vign_i++) {
		for(int vign_j=0 ; vign_j<12 ; vign_j++) {
			Vec3b pixel = resized_frame->at<Vec3b>(vign_j, vign_i);
			for(int vign_k=0 ; vign_k<3 ; vign_k++) {
				returned_unit.image_12[indice] = pixel[vign_k]/255.0;
				indice++;
			}
		}
	}
	return returned_unit;
}

vector<database_unit_training> convert_database_unit_rect(vector<database_unit_training> database, Mat* image, Session* VIS_12_RECT)
{
	vector<database_unit_training> returned_vector;
	for(int i=0 ; i<database.size() ; i++) {
		Point translation = get_translation_vis12rect_vign(database.at(i).image_12, VIS_12_RECT);
		Rect translated_bounding_box = database.at(i).original + translation;
		Mat resized_frame;
		resize((*image)(translated_bounding_box), resized_frame, Size(12, 12), 0, 0, INTER_LINEAR);
		database_unit_training translated_unit = create_unit_12x12(&resized_frame);
		translated_unit.original = translated_bounding_box;
		returned_vector.push_back(translated_unit);
	}
	return returned_vector;
}

void vign_to_12x12(database_unit_training* unit, Mat* resized_frame, Rect bounding_box)
{
	int indice = 0;
	for(int vign_i=0 ; vign_i<12 ; vign_i++) {
		for(int vign_j=0 ; vign_j<12 ; vign_j++) {
			Vec3b pixel = resized_frame->at<Vec3b>(vign_j, vign_i);
			for(int vign_k=0 ; vign_k<3 ; vign_k++) {
				unit->image_12[indice] = pixel[vign_k]/255.0;
				indice++;
			}
		}
	}
}

vector<database_unit_training> resized_frame_to_database_54x36(Mat* resized_frame, Rect bounding_box)
{
	vector<database_unit_training> returned_vector;

	int channels = resized_frame->channels();
	int rows = resized_frame->rows;
	int cols = resized_frame->cols;
	for(int frame_i=0 ; frame_i<15 ; frame_i++) {
		for(int frame_j=0 ; frame_j<9 ; frame_j++) {
			database_unit_training vignette;
			int indice = 0;
			for(int vign_i=0 ; vign_i<12 ; vign_i++) {
				for(int vign_j=0 ; vign_j<12 ; vign_j++) {
					Vec3b pixel = resized_frame->at<Vec3b>(frame_j*3+vign_j, frame_i*3+vign_i);
					for(int vign_k=0 ; vign_k<3 ; vign_k++) {
						vignette.image_12[indice] = pixel[vign_k]/255.0;
						indice++;
					}
				}
			}
			Rect crop_12(frame_i*3, frame_j*3, 12, 12);
			vignette.is_face = overlap_ratio(bounding_box, crop_12);

			vignette.dx = ((bounding_box.x+(bounding_box.width/2.0)) - ((frame_i*3)+(12/2.0)))/(float)12;
			vignette.dy = ((bounding_box.y+(bounding_box.height/2.0)) - ((frame_j*3)+(12/2.0)))/(float)12;
			vignette.dw = (float)(bounding_box.width - 12)/(float)12;
			vignette.dh = (float)(bounding_box.height - 12)/(float)12;
			double xRatio = IMG_X/54.0;
			double yRatio = IMG_Y/36.0;
			vignette.original = Rect((frame_i*3)*xRatio, (frame_j*3)*yRatio, 12*xRatio, 12*yRatio);
			returned_vector.push_back(vignette);
		}
	}
	return returned_vector;
}

vector<database_unit_training> resized_frame_to_field_54x36(Mat* resized_frame)
{
	vector<database_unit_training> returned_vector;
	int channels = resized_frame->channels();
	int rows = resized_frame->rows;
	int cols = resized_frame->cols;
	for(int frame_i=0 ; frame_i<15 ; frame_i++) {
		for(int frame_j=0 ; frame_j<9 ; frame_j++) {
			database_unit_training vignette;
			int indice = 0;
			for(int vign_i=0 ; vign_i<12 ; vign_i++) {
				for(int vign_j=0 ; vign_j<12 ; vign_j++) {
					Vec3b pixel = resized_frame->at<Vec3b>(frame_j*3+vign_j, frame_i*3+vign_i);
					for(int vign_k=0 ; vign_k<3 ; vign_k++) {
						vignette.image_12[indice] = pixel[vign_k]/255.0;
						//cout << vignette.image_12[indice] << endl;
						indice++;
					}
				}
			}
			double xRatio = IMG_X/54.0;
			double yRatio = IMG_Y/36.0;
			vignette.original = Rect((frame_i*3)*xRatio, (frame_j*3)*yRatio, 12*xRatio, 12*yRatio);
			returned_vector.push_back(vignette);
		}
	}
	return returned_vector;
}

Session* CreateSession(string graph_definition_path)
{
	Session* session;
    GraphDef graph_def;
    SessionOptions opts;
    TF_CHECK_OK(ReadBinaryProto(Env::Default(), graph_definition_path, &graph_def));
    TF_CHECK_OK(NewSession(opts, &session));
    TF_CHECK_OK(session->Create(graph_def));
	TF_CHECK_OK(session->Run({}, {}, {"init_all_vars_op"}, nullptr));
	return session;
}



float visage_rate_vis12bb_vign(float image_12[12*12*3], Session* VIS_12_BB)
{
	Tensor network_inputs(DT_FLOAT, TensorShape({BATCH_SIZE, 12* 12* 3}));
	Tensor network_outputs(DT_FLOAT, TensorShape({BATCH_SIZE, 2}));
	std::vector<Tensor> outputs;
	auto _InputTensor = network_inputs.tensor<float, 2>();
	auto _OutputTensor = network_outputs.tensor<float, 2>();

	memcpy(&_InputTensor(0, 0), image_12, 12*12*3*sizeof(float));
	//cout << _InputTensor(0, 0) << endl;
	TF_CHECK_OK(VIS_12_BB->Run({{"x", network_inputs}, {"y", network_outputs}}, {"y_out"}, {}, &outputs));
	auto true_output = outputs[0].tensor<float, 2>();
	return (true_output(0,1)-true_output(0,0));
}

Size get_resize_vis12size_vign(float image_12[12*12*3], Session* VIS_12_SIZE)
{
	Tensor network_inputs(DT_FLOAT, TensorShape({BATCH_SIZE, 12* 12* 3}));
	Tensor network_outputs(DT_FLOAT, TensorShape({BATCH_SIZE, 2}));
	std::vector<Tensor> outputs;
	auto _InputTensor = network_inputs.tensor<float, 2>();
	auto _OutputTensor = network_outputs.tensor<float, 2>();

	memcpy(&_InputTensor(0, 0), image_12, 12*12*3*sizeof(float));
	TF_CHECK_OK(VIS_12_SIZE->Run({{"x", network_inputs}, {"y", network_outputs}}, {"y_out"}, {}, &outputs));
	auto true_output = outputs[0].tensor<float, 2>();

	Size resize(true_output(0,0) * SIZE_SQUARE, true_output(0,1) * SIZE_SQUARE);
	return resize;
}

void show_resize_vis12size(Mat image, Session* VIS_12BB, Session* VIS_12_RECT, Session* VIS_12_SIZE) {
	Mat resized_frame;
	resize(image, resized_frame, Size(54, 36), 0, 0, INTER_LINEAR);
	vector<database_unit_training> vignettes = resized_frame_to_field_54x36(&resized_frame);
	vector<database_unit_training> updated_vignettes;

	vector<Rect> Updated_bounding_boxes;

	for(int i=0 ; i<vignettes.size() ; i++) {
		vignettes.at(i).confidence = visage_rate_vis12bb_vign(vignettes.at(i).image_12, VIS_12BB);
		if(vignettes.at(i).confidence > 0) {
			rectangle(image, vignettes.at(i).original, Scalar( 255, 255, 0 ), 1, 8, 0 );
			Point translation = get_translation_vis12rect_vign(vignettes.at(i).image_12, VIS_12_RECT);
			//Point center = Point(SIZE_SQUARE/2.0, SIZE_SQUARE/2.0);
			//Point originalul = Point(vignettes.at(i).original.x, vignettes.at(i).original.y);
			//circle(image, originalul+center+translation, 0, Scalar( 255, 0, 255 ), 5, 8, 0);
			Rect new_BB = (vignettes.at(i).original + translation);
			Mat new_resized_frame;
			resize(image(new_BB), new_resized_frame, Size(12, 12), 0, 0, INTER_LINEAR);
			database_unit_training new_vignette = create_unit_12x12(&new_resized_frame);
			


			Size resize = get_resize_vis12size_vign(new_vignette.image_12, VIS_12_SIZE);
			cout << "new_BB.x = " << new_BB.x << endl;
			new_BB += resize;
			new_BB.x -= resize.width/2.0;
			new_BB.y -= resize.height/2.0;
			
			new_vignette.original = new_BB;
			updated_vignettes.push_back(new_vignette);
			Updated_bounding_boxes.push_back(new_BB);
		}
	}

	
	for(float prec = 1.0 ; prec>0.0 ; prec-=0.1) {
		Mat image_copy = image.clone();
		vector<Rect> Fuzed_bounding_boxes = perfect_fuze_bounding_boxes(Updated_bounding_boxes, prec);
		for(int i=0 ; i<Fuzed_bounding_boxes.size() ; i++) {
			rectangle(image_copy, Fuzed_bounding_boxes.at(i), Scalar( 255, 0, 255 ), 1, 8, 0 );
		}
		imshow( "test", image_copy );
		waitKey(200);
	}
}

void show_DEMO(Mat image, Session* VIS_12BB, Session* VIS_12_RECT, Session* VIS_12_SIZE, int number) {
	VideoWriter outputvid("../DEMO/Face_detection_DEMO_" + to_string(number) + ".avi", VideoWriter::fourcc('D','I','V','X'), 30, Size(IMG_X, IMG_Y), true);
	for(int i=0 ; i<30 ; i++) {
		outputvid << image;
	}
	imshow( "test", image );
	waitKey(1000);
	Rect Fenetre(0, 0, image.cols, image.rows);
	Mat image_copy_1 = image.clone();
	Mat image_copy_2 = image.clone();
	Mat resized_frame;
	resize(image, resized_frame, Size(54, 36), 0, 0, INTER_LINEAR);
	vector<database_unit_training> vignettes_phase_1 = resized_frame_to_field_54x36(&resized_frame);
	vector<database_unit_training> vignettes_phase_2;
	vector<database_unit_training> vignettes_phase_3;
	vector<Rect> vignettes_phase_4;



	//PHASE 1
	for(int i=0 ; i<vignettes_phase_1.size() ; i++) {
		putText(image_copy_1, "Phase 1 : Face detection", Point(10,20), FONT_HERSHEY_PLAIN, 1 ,Scalar( 255, 255, 255 ), 1,8,false);
		image_copy_2 = image_copy_1.clone();
		vignettes_phase_1.at(i).confidence = visage_rate_vis12bb_vign(vignettes_phase_1.at(i).image_12, VIS_12BB);
		rectangle(image_copy_2, vignettes_phase_1.at(i).original, Scalar( 125, 0, 125 ), 2, 8, 0 );
		
		if(vignettes_phase_1.at(i).confidence > 0) {
			rectangle(image_copy_1, vignettes_phase_1.at(i).original, Scalar( 255, 0, 255 ), 2, 8, 0 );
			vignettes_phase_2.push_back(vignettes_phase_1.at(i));
		}
		outputvid << image_copy_2;
		imshow( "test", image_copy_2 );
		waitKey(33);
	}
	for(int i=0 ; i<15 ; i++) {
		outputvid << image_copy_2;
	}
	waitKey(500);
	
	//PHASE 2
	for(int i=0 ; i<vignettes_phase_2.size() ; i++) {
		image_copy_1 = image.clone();
		putText(image_copy_1, "Phase 2 : Shifting", Point(10,20), FONT_HERSHEY_PLAIN, 1 ,Scalar( 255, 255, 255 ), 1,8,false);
		for(int j=i+1 ; j<vignettes_phase_2.size() ; j++) {
			rectangle(image_copy_1, vignettes_phase_2.at(j).original, Scalar( 255, 0, 255 ), 2, 8, 0 );
		}
		Point center(vignettes_phase_2.at(i).original.x + vignettes_phase_2.at(i).original.width/2, vignettes_phase_2.at(i).original.y + vignettes_phase_2.at(i).original.height/2);
		Point translation = get_translation_vis12rect_vign(vignettes_phase_2.at(i).image_12, VIS_12_RECT);
		Rect new_BB = (vignettes_phase_2.at(i).original + translation);
		new_BB = Fenetre & new_BB;
		Mat new_resized_frame;
		resize(image(new_BB), new_resized_frame, Size(12, 12), 0, 0, INTER_LINEAR);
		database_unit_training new_vignette = create_unit_12x12(&new_resized_frame);
		new_vignette.original = new_BB;
		vignettes_phase_3.push_back(new_vignette);
		line(image_copy_1, center, center+translation, Scalar( 255, 255, 255 ), 2, 8, 0 );
		for(int j=0 ; j<vignettes_phase_3.size() ; j++) {
			rectangle(image_copy_1, vignettes_phase_3.at(j).original, Scalar( 255, 255, 0 ), 2, 8, 0 );
		}
		outputvid << image_copy_1;
		imshow( "test", image_copy_1 );
		waitKey(33);
	}

	for(int i=0 ; i<15 ; i++) {
		outputvid << image_copy_1;
	}
	waitKey(500);
	
	//PHASE 3
	for(int i=0 ; i<vignettes_phase_3.size() ; i++) {
		image_copy_1 = image.clone();
		putText(image_copy_1, "Phase 3 : Resizing", Point(10,20), FONT_HERSHEY_PLAIN, 1 ,Scalar( 255, 255, 255 ), 1,8,false);
		for(int j=i+1 ; j<vignettes_phase_3.size() ; j++) {
			rectangle(image_copy_1, vignettes_phase_3.at(j).original, Scalar( 255, 255, 0 ), 2, 8, 0 );
		}
		Size resize = get_resize_vis12size_vign(vignettes_phase_3.at(i).image_12, VIS_12_SIZE);
		Rect new_BB = (vignettes_phase_3.at(i).original);
		new_BB += resize;
		
		new_BB.x -= resize.width/2.0;
		new_BB.y -= resize.height/2.0;
		vignettes_phase_4.push_back(new_BB);
		for(int j=0 ; j<vignettes_phase_4.size() ; j++) {
			rectangle(image_copy_1, vignettes_phase_4.at(j), Scalar( 0, 255, 255 ), 2, 8, 0 );
		}
		outputvid << image_copy_1;
		imshow( "test", image_copy_1 );
		waitKey(33);
	}

	for(int i=0 ; i<15 ; i++) {
		outputvid << image_copy_1;
	}
	waitKey(500);
	
	//PHASE 4
	for(float prec = 1.0 ; prec>0.5 ; prec-=0.05) {
		image_copy_1 = image.clone();
		putText(image_copy_1, "Phase 4 : Bounding box extraction", Point(10,20), FONT_HERSHEY_PLAIN, 1 ,Scalar( 255, 255, 255 ), 1,8,false);
		vector<Rect> Fuzed_bounding_boxes = perfect_fuze_bounding_boxes(vignettes_phase_4, prec);
		for(int i=0 ; i<Fuzed_bounding_boxes.size() ; i++) {
			rectangle(image_copy_1, Fuzed_bounding_boxes.at(i), Scalar( 0, 255, 255 ), 2, 8, 0 );
		}
		for(int i=0 ; i<10 ; i++) {
			outputvid << image_copy_1;
		}
		imshow( "test", image_copy_1 );
		waitKey(200);
	}
	for(int i=0 ; i<60 ; i++) {
		outputvid << image_copy_1;
	}
	waitKey(1000);
}

int main( int argc, const char** argv )
{

	srand(time(NULL));
	vector<string> image_names;
	vector<Rect> bounding_boxes;
	vector<database_unit_training> database;
	vector<database_unit_training> rectified_database;
	ifstream bb_file ("../Data/Annotations/bounding_boxes.txt", ios::in);
    if (bb_file.fail()){perror(("Can't find ../Data/Annotations/bounding_boxes.txt"));return 0;}
	string data_string;
	bb_file >> data_string;
	while(!bb_file.eof())
	{
		string image_name;
		int Xmin, Ymin, Xmax, Ymax;
		bb_file >> image_name;
		bb_file >> Xmin;
		bb_file >> Ymin;
		bb_file >> Xmax;
		bb_file >> Ymax;
		image_names.push_back(image_name);
		bounding_boxes.push_back(Rect(Xmin, Ymin, Xmax-Xmin, Ymax-Ymin));
	}
	bb_file.close();
	
	
	
	if(READ_DATABASE == 1) {
		//for(int index = 0 ; index<image_names.size() ; index++)
		for(int index = 0 ; index<NUMBER_OF_IMAGES ; index++)
		{
			cout << image_names.at(index) << endl;
			Mat frame = imread( "../Data/Images/" + image_names.at(index), 1 );
			int rows = frame.rows;
			int cols = frame.cols;
			float xRatio = 54.0/IMG_X;
			float yRatio = 36.0/IMG_Y;
			Rect resized_bounding_box(bounding_boxes.at(index).x*xRatio, bounding_boxes.at(index).y*yRatio, bounding_boxes.at(index).width*xRatio, bounding_boxes.at(index).height*yRatio);
			Mat resized_frame;
			resize(frame, resized_frame, Size(54, 36), 0, 0, INTER_LINEAR);
			vector<database_unit_training> database_temp = resized_frame_to_database_54x36(&resized_frame, resized_bounding_box);
			database.insert(database.end(), database_temp.begin(), database_temp.end());
		}
		for(int i=0 ; i<database.size() ; i++) {
			if(database.at(i).is_face == 1) rectified_database.push_back(database.at(i));
		}
		random_shuffle(rectified_database.begin(), rectified_database.end());
		random_shuffle(database.begin(), database.end());
	}
	Session* VIS_12_BB = CreateSession("../Networks/visage_12_bb.pb");
	Session* VIS_12_RECT = CreateSession("../Networks/visage_12_rect.pb");
	Session* VIS_12_SIZE = CreateSession("../Networks/visage_12_size.pb");
	Tensor network_inputs_vis12bb(DT_FLOAT, TensorShape({BATCH_SIZE, 12* 12* 3}));
	Tensor network_inputs_vis12rect(DT_FLOAT, TensorShape({BATCH_SIZE, 12* 12* 3}));
	Tensor network_inputs_vis12size(DT_FLOAT, TensorShape({BATCH_SIZE, 12* 12* 3}));
    Tensor network_outputs_vis12bb(DT_FLOAT, TensorShape({BATCH_SIZE, 2}));
	Tensor network_outputs_vis12rect(DT_FLOAT, TensorShape({BATCH_SIZE, 2}));
	Tensor network_outputs_vis12size(DT_FLOAT, TensorShape({BATCH_SIZE, 2}));

	Tensor save_tensor_vis12bb(DT_STRING, TensorShape({1, 1}));
	save_tensor_vis12bb.matrix< std::string >()( 0,0 ) = "../Networks/visage_12_bb";
	Tensor save_tensor_vis12rect(DT_STRING, TensorShape({1, 1}));
	save_tensor_vis12rect.matrix< std::string >()( 0,0 ) = "../Networks/visage_12_rect";
	Tensor save_tensor_vis12size(DT_STRING, TensorShape({1, 1}));
	save_tensor_vis12size.matrix< std::string >()( 0,0 ) = "../Networks/visage_12_size";

	std::vector<Tensor> outputs;
	
	auto _InputTensor_vis12bb = network_inputs_vis12bb.tensor<float, 2>();
	auto _OutputTensor_vis12bb = network_outputs_vis12bb.tensor<float, 2>();
	auto _InputTensor_vis12rect = network_inputs_vis12rect.tensor<float, 2>();
	auto _OutputTensor_vis12rect = network_outputs_vis12rect.tensor<float, 2>();
	auto _InputTensor_vis12size = network_inputs_vis12size.tensor<float, 2>();
	auto _OutputTensor_vis12size = network_outputs_vis12size.tensor<float, 2>();
	
//////////////////////////////////////////////////////////
	if(SAVE_RUN_BB == 1) {
		for (int i = 0; i < NUMBER_OF_ITER; ++i) {
			for(int batch_index = 0 ; batch_index < BATCH_SIZE ; ++batch_index) {
				for(int x_index = 0 ; x_index < 12 ; ++x_index) {
					for(int y_index = 0 ; y_index < 12 ; ++y_index) {
						for(int c_index = 0 ; c_index < 3 ; ++c_index) {
							memcpy(&_InputTensor_vis12bb(batch_index, 0), &database.at((i*BATCH_SIZE + batch_index)%database.size()).image_12[0], 12*12*3*sizeof(float));
						}
					}
				}
				if(database.at((i*BATCH_SIZE + batch_index)%database.size()).is_face) {
					_OutputTensor_vis12bb(batch_index, 0) = 0;
					_OutputTensor_vis12bb(batch_index, 1) = 1;
				} else {
					_OutputTensor_vis12bb(batch_index, 0) = 1;
					_OutputTensor_vis12bb(batch_index, 1) = 0;
				}
			}
		
			TF_CHECK_OK(VIS_12_BB->Run({{"x", network_inputs_vis12bb}, {"y", network_outputs_vis12bb}}, {}, {"train"}, nullptr)); // Train
		
			if(i%1000 == 0) 
			{
				cout << "iter = " << i << endl;
				TF_CHECK_OK(VIS_12_BB->Run({{"x", network_inputs_vis12bb}, {"y", network_outputs_vis12bb}}, {"loss"}, {}, &outputs));
				auto true_output = outputs[0].tensor<float, 0>();
				cout << "loss = " << true_output << endl;
			}
		}
		TF_CHECK_OK(VIS_12_BB->Run({ { "save/Const:0", save_tensor_vis12bb } }, {}, {"save/control_dependency"}, nullptr)); // Train
	} else {
		TF_CHECK_OK(VIS_12_BB->Run({ { "save/Const:0", save_tensor_vis12bb } }, {}, {"save/restore_all"}, nullptr)); // Train
	}

	
//////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////
	if(SAVE_RUN_RECT == 1) {
		for (int i = 0; i < NUMBER_OF_ITER; ++i) {
			for(int batch_index = 0 ; batch_index < BATCH_SIZE ; ++batch_index) {
				for(int x_index = 0 ; x_index < 12 ; ++x_index) {
					for(int y_index = 0 ; y_index < 12 ; ++y_index) {
						for(int c_index = 0 ; c_index < 3 ; ++c_index) {
							memcpy(&_InputTensor_vis12rect(batch_index, 0), &rectified_database.at((i*BATCH_SIZE + batch_index)%rectified_database.size()).image_12[0], 12*12*3*sizeof(float));
						}
					}
				}
				_OutputTensor_vis12rect(batch_index, 0) = rectified_database.at((i*BATCH_SIZE + batch_index)%rectified_database.size()).dx;
				_OutputTensor_vis12rect(batch_index, 1) = rectified_database.at((i*BATCH_SIZE + batch_index)%rectified_database.size()).dy;
			}
		
			TF_CHECK_OK(VIS_12_RECT->Run({{"x", network_inputs_vis12rect}, {"y", network_outputs_vis12rect}}, {}, {"train"}, nullptr)); // Train

			if(i%1000 == 0) 
			{
				cout << "iter = " << i << endl;
				TF_CHECK_OK(VIS_12_RECT->Run({{"x", network_inputs_vis12rect}, {"y", network_outputs_vis12rect}}, {"loss"}, {}, &outputs));
				auto true_output = outputs[0].tensor<float, 0>();
				cout << "loss = " << true_output << endl;
			}
		}
		TF_CHECK_OK(VIS_12_RECT->Run({ { "save/Const:0", save_tensor_vis12rect } }, {}, {"save/control_dependency"}, nullptr)); // Train
	} else {
		TF_CHECK_OK(VIS_12_RECT->Run({ { "save/Const:0", save_tensor_vis12rect } }, {}, {"save/restore_all"}, nullptr)); // Train
	}
//////////////////////////////////////////////////////////



//////////////////////////////////////////////////////////
	if(SAVE_RUN_SIZE == 1) {
		vector<database_unit_training> translated_database;
		for(int index = 0 ; index<NUMBER_OF_IMAGES ; index++)
		{
			cout << image_names.at(index) << endl;
			Mat frame = imread( "../Data/Images/" + image_names.at(index), 1 );
			int rows = frame.rows;
			int cols = frame.cols;
			float xRatio = 54.0/IMG_X;
			float yRatio = 36.0/IMG_Y;
			Rect resized_bounding_box(bounding_boxes.at(index).x*xRatio, bounding_boxes.at(index).y*yRatio, bounding_boxes.at(index).width*xRatio, bounding_boxes.at(index).height*yRatio);
			Mat resized_frame;
			resize(frame, resized_frame, Size(54, 36), 0, 0, INTER_LINEAR);
			vector<database_unit_training> database_temp = resized_frame_to_database_54x36(&resized_frame, resized_bounding_box);
			for(int i=0 ; i<database_temp.size() ; i++) {
				if(database_temp.at(i).is_face == 1) {
					Point translation = get_translation_vis12rect_vign(database_temp.at(i).image_12, VIS_12_RECT);
				
					Rect new_vign = database_temp.at(i).original + translation;
					if(new_vign.x > 0 && new_vign.y > 0 && new_vign.x+new_vign.width < IMG_X && new_vign.y+new_vign.height < IMG_Y) {
						Mat resized_frame;
						resize(frame(new_vign), resized_frame, Size(12, 12), 0, 0, INTER_LINEAR);
						database_unit_training temp_unit;
						vign_to_12x12(&temp_unit, &resized_frame, new_vign);
						temp_unit.is_face = 1;
						temp_unit.dx = database_temp.at(i).dx - translation.x;
						temp_unit.dy = database_temp.at(i).dy - translation.y;
						temp_unit.dw = database_temp.at(i).dw;
						temp_unit.dh = database_temp.at(i).dh;
						translated_database.push_back(temp_unit);
					}
				}
			}
		}
		for (int i = 0; i < NUMBER_OF_ITER; ++i) {
			for(int batch_index = 0 ; batch_index < BATCH_SIZE ; ++batch_index) {
				for(int x_index = 0 ; x_index < 12 ; ++x_index) {
					for(int y_index = 0 ; y_index < 12 ; ++y_index) {
						for(int c_index = 0 ; c_index < 3 ; ++c_index) {
							memcpy(&_InputTensor_vis12size(batch_index, 0), &translated_database.at((i*BATCH_SIZE + batch_index)%translated_database.size()).image_12[0], 12*12*3*sizeof(float));
						}
					}
				}
				_OutputTensor_vis12size(batch_index, 0) = translated_database.at((i*BATCH_SIZE + batch_index)%translated_database.size()).dw;
				_OutputTensor_vis12size(batch_index, 1) = translated_database.at((i*BATCH_SIZE + batch_index)%translated_database.size()).dh;
			}
		
			TF_CHECK_OK(VIS_12_SIZE->Run({{"x", network_inputs_vis12size}, {"y", network_outputs_vis12size}}, {}, {"train"}, nullptr)); // Train

			if(i%1000 == 0) 
			{
				cout << "iter = " << i << endl;
				TF_CHECK_OK(VIS_12_SIZE->Run({{"x", network_inputs_vis12size}, {"y", network_outputs_vis12size}}, {"loss"}, {}, &outputs));
				auto true_output = outputs[0].tensor<float, 0>();
				cout << "loss = " << true_output << endl;
			}
		}
		TF_CHECK_OK(VIS_12_SIZE->Run({ { "save/Const:0", save_tensor_vis12size } }, {}, {"save/control_dependency"}, nullptr)); // Train
	} else {
		TF_CHECK_OK(VIS_12_SIZE->Run({ { "save/Const:0", save_tensor_vis12size } }, {}, {"save/restore_all"}, nullptr)); // Train
	}
//////////////////////////////////////////////////////////


	for(int i=0 ; i<image_names.size() ; i++) {
		Mat image = imread( "../Data/Images/" + image_names.at(i), 1 );
		show_DEMO(image, VIS_12_BB, VIS_12_RECT, VIS_12_SIZE, i);
		//waitKey(10000);
	}
	
	
	
    VIS_12_BB->Close();
	VIS_12_RECT->Close();
	VIS_12_SIZE->Close();
    delete VIS_12_BB;
	delete VIS_12_RECT;
	delete VIS_12_SIZE;
	return 0;
}

