#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <ctime>

using namespace std;
using namespace cv;

class association_class {
	public:
	float strength;
	Rect aabb1;
	Rect aabb2;
	int index1;
	int index2;
	association_class(float _strength, Rect _aabb1, Rect _aabb2, int _index1, int _index2);
};

class case_class {
	public:
	Rect aabb;
	int bond;
};

float calc_similarity(Rect aabb1, Rect aabb2);

vector<Rect> perfect_fuze_bounding_boxes(vector<Rect> Bounding_boxes, float precision);
vector<Rect> approximate_fuze_bounding_boxes(vector<Rect> Bounding_boxes, float precision);
