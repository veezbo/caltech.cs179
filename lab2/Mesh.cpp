/*
 *  Mesh.cpp
 *  lab1
 *
 *  Created by William Clark on 1/22/07.
 *
 */

#include "Mesh.h"
#include <string.h>
Mesh::Mesh()
{
    initialized = false;
}

bool Mesh::parseOBJ(const char *filename)
{
    cout << "Attempting to load mesh from .obj file: " << filename << endl;
    if (initialized)
    {
        cerr << "Mesh already loaded." << endl;
        return false;
    }
    
    ifstream file(filename);
    
    if (! file)
    {
        cerr << "Unable to open input file: "
        << filename << "." << endl;
        return false;
    }
    
    char buffer[200];
    
    bool finishedWithVerts = false;
    while (!finishedWithVerts)
    {
        if (file.eof())
        {
            cerr << "Abrupt end of file" << endl;
            return false;
        }
        file.getline(buffer, 200);
        if (buffer[0] != 'v')
        {
            if (buffer[0] == 'f')
            {
                finishedWithVerts = true;
            }
            else
            {
                cerr << "Invalid line while reading verts: " << buffer << endl;
                return false;
            }
        }
        else
        {
            float x, y, z;
            sscanf(buffer, "v %f %f %f", &x, &y, &z);
            Vector3 pos(x, y, z);
            vertex.push_back(Vertex(pos));
        }
    }
    
    int vsize = vertex.size();
    
    while (!file.eof())
    {
        int a, b, c, d;
        int vertsInFace = sscanf(buffer, "f %d %d %d %d", &a, &b, &c, &d);
        a -= 1, b -= 1, c -= 1, d -= 1;
        if (vertsInFace < 3 || vertsInFace > 4)
        {
            cerr << "Face has invalid number of vertices: " << buffer << endl;
            return false;
        }
        if (a >= vsize || b >= vsize || c >= vsize || (vertsInFace == 4 && d >= vsize))
        {
            cerr << "Face references invalid vertices: " << buffer << endl;
            return false;
        }
        Vertex* verts[3];
        verts[0] = &vertex[a];
        verts[1] = &vertex[b];
        verts[2] = &vertex[c];
        face.push_back(Triangle(verts));
        if (vertsInFace == 4)
        {
            verts[0] = &vertex[c];
            verts[1] = &vertex[d];
            verts[2] = &vertex[a];
            face.push_back(Triangle(verts));
        }
        file.getline(buffer, 200);
    }
    
    processData();
    initialized = true;
    
    cout << "Mesh loaded successfully." << endl;
    return true;
}

bool Mesh::parseOFF(const char *filename)
{
    cout << "Attempting to load mesh from .off file: " << filename << endl;
    if (initialized)
    {
        cerr << "Mesh already loaded." << endl;
        return false;
    }
    
    ifstream file(filename);
    
    if (! file)
    {
        cerr << "Unable to open input file: "
        << filename << "." << endl;
        return false;
    }
    
    char buffer[200];
    
    // Check for OFF header.
    file.getline(buffer, 200);
    char *header = "OFF";
    if (strcmp(buffer, header) != 0)
    {
        cerr << "Invalid OFF file header: " << buffer << endl;
        return false;
    }
    
    // Read in the size of the mesh.
    file.getline(buffer, 200);
    int vsize, fsize, esize;
    sscanf(buffer, "%d %d %d", &vsize, &fsize, &esize);
    if (vsize < 0 || fsize < 0)
    {
        cerr << "Invalid mesh size: " << buffer << endl;
        return false;
    }
    
    // Read in the vertices.
    for (int i = 0; i < vsize; i++)
    {
        if (file.eof())
        {
            cerr << "Abrupt end of file" << endl;
            return false;
        }
        file.getline(buffer, 200);
        float x, y, z;
        sscanf(buffer, "%f %f %f", &x, &y, &z);
        Vector3 pos(x, y, z);
        vertex.push_back(Vertex(pos));
    }
    
    // Read in the faces.
    for (int i = 0; i < fsize; i++)
    {
        if (file.eof())
        {
            cerr << "Abrupt end of file" << endl;
            return false;
        }
        file.getline(buffer, 200);
        int a, b, c;
        sscanf(buffer, "3 %d %d %d", &a, &b, &c);
        if (a >= vsize || b >= vsize || c >= vsize)
        {
            cerr << "Face references invalid vertices: " << buffer << endl;
            return false;
        }
        Vertex* verts[3];
        verts[0] = &vertex[a];
        verts[1] = &vertex[b];
        verts[2] = &vertex[c];
        face.push_back(Triangle(verts));
    }
    
    processData();
    
    cout << "Mesh loaded successfully." << endl;
    initialized = true;
    return true;
}

void Mesh::processData()
{
    // All we really need to do is normalize the vertex normals, as the
    // faces have been adding into them.
    for(int i = 0; i < vertex.size(); i++)
    {
        vertex[i].normal.normalize();
    }
}
