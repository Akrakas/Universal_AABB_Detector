#include "Rect_fuser.h"

association_class::association_class(float _strength, Rect _aabb1, Rect _aabb2, int _index1, int _index2)
{
	strength = _strength;
	aabb1 = _aabb1;
	aabb2 = _aabb2;
	index1 = _index1;
	index2 = _index2;
}

float calc_similarity(Rect aabb1, Rect aabb2){
	Rect common_area = aabb1 & aabb2;
	float common_area_surface = common_area.width*common_area.height;
	float aabb1_surface = aabb1.width*aabb1.height;
	float aabb2_surface = aabb2.width*aabb2.height;
	float similarity = common_area_surface/aabb1_surface + common_area_surface/aabb1_surface;
	return similarity;
}

//Take longer than an approximate
vector<Rect> perfect_fuze_bounding_boxes(vector<Rect> Bounding_boxes, float precision)
{
	//SANITY CHECK
	if(precision < 0.1) precision = 0.1;
	if(precision > 1.0) precision = 1.0;
	
	vector<Rect> Fuzed_Bounding_boxes;
	vector<case_class> Bounding_boxes_cases;

	for(int i=0 ; i<Bounding_boxes.size() ; i++) {
		case_class temp_case;
		temp_case.aabb = Bounding_boxes.at(i);
		temp_case.bond = 1;
		Bounding_boxes_cases.push_back(temp_case);
	}

	//Parsing of the boxes and their bond
	bool done = false;
	while(!done) {
		vector<association_class> associations;
		for(int i=0 ; i<Bounding_boxes_cases.size() ; i++) {
			for(int j=i+1 ; j<Bounding_boxes_cases.size() ; j++) {
				float similarity = calc_similarity(Bounding_boxes_cases.at(i).aabb, Bounding_boxes_cases.at(j).aabb);
				association_class temp_associations(similarity, Bounding_boxes_cases.at(i).aabb, Bounding_boxes_cases.at(j).aabb, i, j);
				associations.push_back(temp_associations);
			}
		}
		//Start fuzing here	
		int strongest_connexion_index = -1;
		float strongest_connexion = 0.0;
		for(int i=0 ; i<associations.size() ; i++) {
			if(associations.at(i).strength > strongest_connexion) {
				strongest_connexion_index = i;
				strongest_connexion = associations.at(i).strength;
			}
		}
		if(strongest_connexion < (2*precision)) done = true;
		else {
			Rect aabb1 = Bounding_boxes_cases.at(associations.at(strongest_connexion_index).index1).aabb;
			Rect aabb2 = Bounding_boxes_cases.at(associations.at(strongest_connexion_index).index2).aabb;
			int bond1 = Bounding_boxes_cases.at(associations.at(strongest_connexion_index).index1).bond;
			int bond2 = Bounding_boxes_cases.at(associations.at(strongest_connexion_index).index2).bond;
			int somme_bonds = bond1+bond2;

			Point temp_point(((aabb1.x*bond1)+(aabb2.x*bond2))/somme_bonds, ((aabb1.y*bond1)+(aabb2.y*bond2))/somme_bonds);
			Size temp_size(((aabb1.width*bond1)+(aabb2.width*bond2))/somme_bonds, ((aabb1.height*bond1)+(aabb2.height*bond2))/somme_bonds);
			Rect fuzed_bounding_box = Rect(temp_point, temp_size);
			Bounding_boxes_cases.erase(Bounding_boxes_cases.begin() + associations.at(strongest_connexion_index).index2);
			Bounding_boxes_cases.erase(Bounding_boxes_cases.begin() + associations.at(strongest_connexion_index).index1);
			case_class temp_case;
			temp_case.aabb = fuzed_bounding_box;
			temp_case.bond = somme_bonds;
			Bounding_boxes_cases.push_back(temp_case);
		}
	}
	for(int i=0 ; i<Bounding_boxes_cases.size() ; i++) {
		Fuzed_Bounding_boxes.push_back(Bounding_boxes_cases.at(i).aabb);
	}
	return Fuzed_Bounding_boxes;
}

vector<Rect> approximate_fuze_bounding_boxes(vector<Rect> Bounding_boxes, float precision)
{
	//TBD
	vector<Rect> Fuzed_Bounding_boxes;
	return Fuzed_Bounding_boxes;
}
