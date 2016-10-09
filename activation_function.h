#include <cmath>

/*range from 0 to 1
 */
double sigmoid(double x)
{
    return 1/(1+exp(-1*x));
}


/*range from -1 to 1
 */
double sigmoid_h(double x)
{
    return 2/(1+exp(-x))-1;
}

/* range from -2 to 2
 */
double sigmoid_g(double x)
{
    return 4/(1+exp(-x))-2;
}
