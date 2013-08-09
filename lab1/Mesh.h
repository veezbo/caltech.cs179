/*
 *  Mesh.h
 *  lab1
 *
 *  Created by William Clark on 1/22/07.
 *
 */

#ifndef MESH_H
#define MESH_H

#include <vector>
#include <fstream>
#include <iostream>
#include "Vector3.h"

using namespace std;

struct Vertex
{
    Vector3 position;
    Vector3 normal;
    Vertex(float x, float y, float z) : position(Vector3(x, y, z)) {}
    Vertex(Vector3 pos) : position(pos) {}
    Vertex(Vector3 pos, Vector3 norm) : position(pos), normal(norm) {}
};

struct Triangle
{
    Vertex* vertex[3];
    Vector3 normal;
    
    Triangle(Vertex* v[3])
    {
        vertex[0] = v[0];
        vertex[1] = v[1];
        vertex[2] = v[2];
        Vector3 e1 = v[1]->position - v[0]->position;
        Vector3 e2 = v[2]->position - v[0]->position;
        normal = e1.cross(e2);
        normal.normalize();
        v[0]->normal += normal;
        v[1]->normal += normal;
        v[2]->normal += normal;
    }
    
    Triangle& operator=(const Triangle& other)
    {
        vertex[0] = other.vertex[0];
        vertex[1] = other.vertex[1];
        vertex[2] = other.vertex[2];
        normal = other.normal;
        return *this;
    }
};

class Mesh
{
public:
    vector<Vertex> vertex;
    vector<Triangle> face;
    
    bool initialized;
    
    Mesh();
    
    bool parseOBJ(const char *filename);
    bool parseOFF(const char *filename);
    
private:
    void processData();
};

#endif // MESH_H
