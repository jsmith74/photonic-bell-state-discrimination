#include <cstdlib>
#include <iostream>
#include <string>
#include <sstream>

int main(){

    for(int i=100000;i<429981696;i+=100000){

        std::stringstream ss;

        ss << i;

        std::string commandLine;

        ss >> commandLine;

        commandLine = "./LinearOpticalSimulation " + commandLine;

        std::system( commandLine.c_str() );

    }

    return 0;

}
