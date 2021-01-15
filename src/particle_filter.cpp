/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <float.h>

#include "helper_functions.h"

#define WEIGHT 1

using std::string;
using std::vector;
using std::default_random_engine;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * DONE: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * DONE: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  num_particles = 50;  // DONE: Set the number of particles
  default_random_engine random_gen;
  
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  for (int id = 0; id < num_particles; id++)
  {
    Particle single_particle(id, dist_x(random_gen), dist_y(random_gen), dist_theta(random_gen), WEIGHT);  
    particles.push_back(single_particle);
    weights.push_back(0);
  }

  is_initialized = true;
  
  //for(int i = 0; i < num_particles; i++)
  //{
  //  std::cout << "INIT - X: " << particles[i].x << " Y: " << particles[i].y << " Theta: " << particles[i].theta << " Weight: " << particles[i].weight << std::endl;
  //}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * DONE: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */  
  default_random_engine random_gen;
  
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);  
  
  if (yaw_rate == 0) 
    yaw_rate = 1;
  
  for (int id = 0; id < num_particles; id++)
  {
    double delta_x = (velocity / yaw_rate) * (sin(particles[id].theta + yaw_rate * delta_t) - sin(particles[id].theta));
    particles[id].x += delta_x + dist_x(random_gen); //Add delta_x + random Gaussian noise
    
    double delta_y = (velocity / yaw_rate) * (cos(particles[id].theta) - cos(particles[id].theta + yaw_rate * delta_t));
    particles[id].y += delta_y + dist_y(random_gen); //Add delta_y + random Gaussian noise
    
    double delta_theta = yaw_rate * delta_t;
    particles[id].theta += delta_theta + dist_theta(random_gen); //Add delta_theta + random Gaussian noise
    
    //std::cout << "Particle -- x: " << particles[id].x << " y: " << particles[id].y << " theta: " << particles[id].theta << std::endl; 
  }
  
  //for(int i = 0; i < num_particles; i++)
  //{
  //  std::cout << "Prediction - X: " << particles[i].x << " Y: " << particles[i].y << " Theta: " << particles[i].theta << " Weight: " << particles[i].weight << std::endl;
  //} 
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * DONE: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (uint i = 0; i < observations.size(); i++) 
  {
    double closest = DBL_MAX;
    for (uint j = 0; j < predicted.size(); j++)
    {
      double dist_val = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (dist_val < closest) 
      {
        observations[i].id = predicted[j].id;
        closest = dist_val;
      }
    }
    //std::cout << "Closest_id: " << observations[i].id << std::endl;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * DONE: Update the weights of each particle using a multi-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int particle_id = 0; particle_id < num_particles; particle_id++)
  {
    vector<LandmarkObs> map_coord_observations;
    for (uint observation_id = 0; observation_id < observations.size(); observation_id++) 
    {
      LandmarkObs map_coord_obs;
      map_coord_obs.x = particles[particle_id].x + (cos(particles[particle_id].theta) * observations[observation_id].x) - (sin(particles[particle_id].theta) * observations[observation_id].y);
      map_coord_obs.y = particles[particle_id].y + (sin(particles[particle_id].theta) * observations[observation_id].x) + (cos(particles[particle_id].theta) * observations[observation_id].y);
      map_coord_obs.id = observation_id;
      map_coord_observations.push_back(map_coord_obs);
      //std::cout << "MAP coord - id: " << map_coord_obs.id << " x: " << map_coord_obs.x << " y: " << map_coord_obs.y << std::endl;
    }
    
    vector<LandmarkObs> predicted_within_range;    
    for (uint map_landmark_id = 0; map_landmark_id < map_landmarks.landmark_list.size(); map_landmark_id++)
    {
      double distance = dist(particles[particle_id].x, particles[particle_id].y, map_landmarks.landmark_list[map_landmark_id].x_f, map_landmarks.landmark_list[map_landmark_id].y_f);
      //std::cout << "Distance: " << distance << std::endl;
      if (distance < sensor_range) 
      {
        LandmarkObs landmarkobs;
        landmarkobs.id = map_landmarks.landmark_list[map_landmark_id].id_i;
        landmarkobs.x = map_landmarks.landmark_list[map_landmark_id].x_f;
        landmarkobs.y = map_landmarks.landmark_list[map_landmark_id].y_f;
        predicted_within_range.push_back(landmarkobs);
        //std::cout << "Landmark in sensor range - id: " << landmark.id << " x: " << landmark.x << " y: " << landmark.y << std::endl;
      }
    }
    
    dataAssociation(predicted_within_range, map_coord_observations);
  
    particles[particle_id].weight = 1;
    double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
    for (uint obs_id = 0; obs_id < map_coord_observations.size(); obs_id++) 
    {
      //multi-variate Gaussian
      for (uint pred_id = 0; pred_id < predicted_within_range.size(); pred_id++)
      {
        if (predicted_within_range[pred_id].id == map_coord_observations[obs_id].id)
        {
          //std::cout << "Found associated!" << std::endl;       
          double exponent = (pow(map_coord_observations[obs_id].x - predicted_within_range[pred_id].x, 2) / (2 * pow(std_landmark[0], 2))) + (pow(map_coord_observations[obs_id].y - predicted_within_range[pred_id].y, 2) / (2 * pow(std_landmark[1], 2)));
          double weight = gauss_norm * exp(-exponent);
    
          particles[particle_id].weight *= weight;
        }
      }
    }
    weights[particle_id] = particles[particle_id].weight;
  }
}

void ParticleFilter::resample() {
  /**
   * DONE: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  default_random_engine random_gen;
  
  double max_weight = *max_element(weights.begin(), weights.end());
  
  normal_distribution<double> dist_double_weight(0.0, max_weight);
  int index = rand() % (num_particles - 1);
  
  double beta = 0;
  
  std::vector<Particle> particles_after_resampling;
  
  for (int particle_id = 0; particle_id < num_particles; particle_id++)
  {
    beta = beta + dist_double_weight(random_gen) * 2;
    while (beta > weights[index])
    {
      beta = beta - weights[index];
      index = (index + 1) % num_particles;
    }  
    particles_after_resampling.push_back(particles[index]);
  }
  particles = particles_after_resampling;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}