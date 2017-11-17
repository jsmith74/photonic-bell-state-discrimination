#include "UGeneration.h"

#define C1 1e-4
#define C2 0.9

#define MACHINEPRECISION 1.1e-16

#define ZOOM_GUARD 50

UGeneration::UGeneration(){



}

void UGeneration::findZeroUnitary(Eigen::VectorXd& position){

    minimize(position);

    return;

}

double UGeneration::f( Eigen::VectorXd& position ){

    int matSize = std::sqrt(position.size());

    Eigen::MatrixXcd U( matSize, matSize );

    setAntiHermitian( U, position );

    U = U.exp().eval();

    double output = 0;

    for(int j=0;j<U.cols();j++) for(int i=0;i<U.rows();i++){

        if( zeroEntries(i,j) == 0 ) output += std::norm( U(i,j) );

    }

    return output;

}

void UGeneration::minimize(Eigen::VectorXd& position){

    double tol = 0.001;

    double denom, rho;

    Eigen::MatrixXd H,I;

    Eigen::VectorXd gradient;

    Eigen::VectorXd s,y;

    gradient.resize( position.size() );

    stepMonitor = f(position);

    setGradient(position,gradient);

    H = (1.0/(gradient.norm())) * Eigen::MatrixXd::Identity(position.size(),position.size());

    I = Eigen::MatrixXd::Identity( position.size() , position.size() );

    alphaPosition.resize(position.size());

    while(true){

        if(gradient.norm() < tol) break;

        p = - H * gradient;

        s = -position;

        y = -gradient;

        position += alpha(position,gradient,p) * p;

        stepMonitor =  f(position);

        if(isnan(stepMonitor)) assert( false && "NAN PROBLEM WITH U");

        setGradient(position,gradient);

        s += position;

        y += gradient;

        denom = y.transpose() * s;

        if(denom == 0.0) break;

        rho = 1.0 / denom;

        H = (I - rho * s * y.transpose()) * H * (I - rho * y * s.transpose()) + rho * s * s.transpose();

    }

    return;

}

double UGeneration::alpha(Eigen::VectorXd& position,Eigen::VectorXd& gradient,Eigen::VectorXd& p){

    double alpha0 = 0.0;
    double alpha1 = 1.0;
    double alpha2 = alphaMax;

    phi0 = stepMonitor;
    phiPrime0 = gradient.transpose() * p;

    double phi1 = phi(position,alpha1);

    if(phi1 > phi0 + C1 * alpha1 * phiPrime0){

        return zoom(position,alpha0,alpha1,phi0,phi1,phiPrime0);

    }

    double phiPrime1 = phiPrime(position,alpha1);

    if(std::abs(phiPrime1) <= -C2 * phiPrime0){

        return alpha1;

    }

    if(phiPrime1 >= 0){

        return zoom(position,alpha1,alpha0,phi1,phi0,phiPrime0);

    }

    double phi2 = phi(position,alpha2);

    if(phi2 > phi0 + C1 * alpha2 * phiPrime0 || phi2 >= phi1){

        return zoom(position,alpha1,alpha2,phi1,phi2,phiPrime1);

    }

    double phiPrime2 = phiPrime(position,alpha2);

    if(std::abs(phiPrime2) <= -C2 * phiPrime0){

        return alpha2;

    }

    if(phiPrime2 >= 0){

        return zoom(position,alpha2,alpha1,phi2,phi1,phiPrime1);

    }

    return alpha2;

}


void UGeneration::setAlphaJ(double& alphaj,double& alphaLow,double& alphaHigh,double& phiLow,double& phiHigh,double& phiLowPrime){

    if(alphaLow < alphaHigh){

        alphaj = alphaHigh * alphaHigh * phiLowPrime - alphaLow * (2.0 * phiHigh - 2.0 * phiLow + alphaLow * phiLowPrime);

        if((-phiHigh + phiLow + (alphaHigh - alphaLow) * phiLowPrime) != 0.0){

            alphaj /= 2.0 * (-phiHigh + phiLow + (alphaHigh - alphaLow) * phiLowPrime);

        }

        else assert(alphaj == 0.0);

        secondDerivativeTest = phiHigh - phiLow + (alphaLow-alphaHigh) * phiLowPrime;

        if((alphaHigh-alphaLow) * (alphaHigh-alphaLow) != 0.0){

            secondDerivativeTest /= (alphaHigh-alphaLow) * (alphaHigh-alphaLow);

        }

        else assert(std::abs(secondDerivativeTest) <= 100 * MACHINEPRECISION);

    }

    else{

        alphaj = alphaLow * alphaLow * phiLowPrime - alphaHigh * (2.0 * phiLow - 2.0 * phiHigh + alphaHigh * phiLowPrime);

        if((-phiLow + phiHigh + (alphaLow - alphaHigh) * phiLowPrime) != 0.0){

            alphaj /= 2.0 * (-phiLow + phiHigh + (alphaLow - alphaHigh) * phiLowPrime);

        }

        //else assert(alphaj == 0.0);

        secondDerivativeTest = phiLow - phiHigh + (alphaHigh-alphaLow) * phiLowPrime;

        if((alphaLow-alphaHigh) * (alphaLow-alphaHigh) != 0.0){

            secondDerivativeTest /= (alphaLow-alphaHigh) * (alphaLow-alphaHigh);

        }

        else assert(std::abs(secondDerivativeTest) <= 100 * MACHINEPRECISION);

    }


    return;

}


double UGeneration::zoom(Eigen::VectorXd& position,double alphaLow,double alphaHigh,double phiLow,double phiHigh,double phiLowPrime){

    double alphaj,phij,phiPrimej;

    int zoomCounter = 0;

    while(true){

        zoomCounter++;

        if(zoomCounter > ZOOM_GUARD) {

            return alphaj;

        }

        setAlphaJ(alphaj,alphaLow,alphaHigh,phiLow,phiHigh,phiLowPrime);

        if(secondDerivativeTest < 0.0){

            if(phiHigh < phi0 + C1 * alphaHigh * phiPrime0){

                return alphaHigh;

            }

            else if(phiLow < phi0 + C1 * alphaLow * phiPrime0){

                return alphaLow;

            }

            else{

                return alphaHigh;

            }

        }

        phij = phi(position,alphaj);

        if(phij > phi0 + C1 * alphaj * phiPrime0 || phij >= phiLow){

            alphaHigh = alphaj;
            phiHigh = phij;

        }

        else{

            phiPrimej = phiPrime(position,alphaj);



            if(std::abs(phiPrimej) <= -C2 * phiPrime0){

                return alphaj;

            }

            if(std::abs(phiPrimej) <= C2 * phiPrime0 && phiPrime0 >0){

                return alphaj;

            }

            if(phiPrimej * (alphaHigh - alphaLow) >= 0.0){

                alphaHigh = alphaLow;
                phiHigh = phiLow;

            }

            alphaLow = alphaj;

            phiLow = phij;

        }

        if(alphaHigh < alphaLow){

            phiLowPrime = phiPrime(position,alphaHigh);

        }

        else{

            phiLowPrime = phiPrime(position,alphaLow);

        }

    }

}

double UGeneration::phiPrime(Eigen::VectorXd& position,double& a){

    double eps = sqrt(MACHINEPRECISION);

    double aPeps = a + eps;
    return (phi(position,aPeps)-phi(position,a))/eps;

}

double UGeneration::phi(Eigen::VectorXd& position,double& a){

    alphaPosition = position + a * p;

    return f(alphaPosition);

}

void UGeneration::setGradient(Eigen::VectorXd& position,Eigen::VectorXd& gradient){

    double eps = sqrt(MACHINEPRECISION);

    for(int i=0;i<position.size();i++){

        gradient(i)  = -stepMonitor;
        position(i) += eps;
        gradient(i) += f(position);
        position(i) -= eps;
        gradient(i) /= eps;

    }

    return;

}


void UGeneration::setAntiHermitian( Eigen::MatrixXcd& H,Eigen::VectorXd& position ){

    int k = 0;

    for(int i=0;i<H.rows();i++) for(int j=i;j<H.cols();j++){

        H(i,j) = position(k);
        H(j,i) = position(k);

        k++;

    }

    std::complex<double> I(0.0,1.0);

    for(int i=0;i<H.rows();i++) for(int j=i+1;j<H.cols();j++){

        H(i,j) *= std::exp( I * position(k) );

        H(j,i) *= std::exp( -I * position(k) );

        k++;

    }

    H = I * H;

    return;

}

void UGeneration::setZeroEntryQuant(Eigen::MatrixXcd& U){

    zeroEntryQuant = 0;

    for(int j=0;j<U.cols();j++) for(int i=0;i<U.rows();i++){

        if(zeroEntries(i,j) == 0) zeroEntryQuant += std::norm( U(i,j) );

    }

    return;

}


void UGeneration::initializeUCondition1(Eigen::MatrixXcd& U){

    int ancillaRows = U.rows() - 4;

    zeroEntries.resize(U.rows(),U.cols());

    initializeZeroEntries();

    setZeroEntriesRandomlyCondition1(ancillaRows);

    while( checkCondition1() ){

        initializeZeroEntries();
        setZeroEntriesRandomlyCondition1(ancillaRows);

    }

    std::cout << "Zero entries of the U matrix:\n" << zeroEntries << std::endl << std::endl;

    return;

}



void UGeneration::setZeroEntriesRandomlyCondition1(int ancillaRows){

    int zeroChoiceRange = ancillaRows + 2;

    for(int j=0;j<zeroEntries.cols();j++){

        int zeroChoices[3];

        zeroChoices[0] = rand() % zeroChoiceRange;

        zeroChoices[1] = rand() % zeroChoiceRange;

        while(zeroChoices[1] == zeroChoices[0]) zeroChoices[1] = rand() % zeroChoiceRange;

        zeroChoices[2] = rand() % zeroChoiceRange;

        while(zeroChoices[2] == zeroChoices[1] || zeroChoices[2] == zeroChoices[0]) zeroChoices[2] = rand() % zeroChoiceRange;

        for(int i=0;i<3;i++){

            if(zeroChoices[i] < ancillaRows) zeroEntries(zeroChoices[i],j) = 0;
            else if(zeroChoices[i]==ancillaRows){
                zeroEntries(ancillaRows,j) = 0;
                zeroEntries(ancillaRows+1,j) = 0;
            }
            else if(zeroChoices[i]==ancillaRows+1){
                zeroEntries(ancillaRows+2,j) = 0;
                zeroEntries(ancillaRows+3,j) = 0;
            }

        }
    }

}

void UGeneration::initializeZeroEntries(){

    for(int i=0;i<zeroEntries.rows();i++) for(int j=0;j<zeroEntries.cols();j++) zeroEntries(i,j) = 1;

    return;

}

bool UGeneration::checkCondition1(){

    for(int j=0;j<zeroEntries.rows();j++){

        if( zeroEntries.row(j).sum() == 0 ) return true;

    }

    for(int j=0;j<zeroEntries.cols();j++){

        Eigen::VectorXi zeroChoices;

        zeroChoices.resize(0);

        int k=0;
        for(int i=0;i<zeroEntries.rows();i++){
            if(zeroEntries(i,j) == 0){
                zeroChoices.conservativeResize( k+1 );
                zeroChoices[k] = i;
                k++;
            }
        }

        for(int i=j+1;i<zeroEntries.cols();i++){

            bool weGood = false;

            for(int l=0;l<zeroChoices.size();l++) if(zeroEntries( zeroChoices(l),i ) == 0) weGood = true;

            if(!weGood) return true;

        }

    }

    return false;

}
