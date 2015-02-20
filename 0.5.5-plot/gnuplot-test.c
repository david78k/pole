#include <stdio.h>
#include <stdlib.h>

#define GNUPLOT "gnuplot -persist"
#define sample_size 500 

FILE *gp;
char *prefix = "180k-test1";
char *fname = "180k-fm200-sup1-sample1-r1.test1";
//char *fname = "180k-fm200-sup1-sample1-r1.train";
//char *prefix = "180k-train";
int sample_period = 100; // 1, 10, 100

//int sample_size = 500;
int lastlines = 180000 - sample_size;
char output[30];

// sample_loc: first -1, last 0, all 1
void plot(int sample_loc, int col) {
	char *colstr;
	char *type = " with lines ";
	switch(col) {
		case 1: colstr = "spikes"; type = ",\\"; break;
		//case 2: colstr = "R"; type = ""; break;
		case 3: colstr = "rhat_0"; break;
		case 4: colstr = "rhat_1"; break;
		case 5: colstr = "theta"; break;
		case 6: colstr = "theta_dot"; break;
		case 7: colstr = "h"; break;
		case 8: colstr = "h_dot"; break;
		case 9: colstr = "force"; break;
		default: break;
	}

	if(sample_loc == -1)
		sprintf(output, "%s-%s-first%d.png", prefix, colstr, sample_size);
	else if (sample_loc == 0)
		sprintf(output, "%s-%s-last%d.png", prefix, colstr, sample_size);
	else
		sprintf(output, "%s-%s.png", prefix, colstr);

	if(col != 2)
	        fprintf(gp, "set output '%s'\n", output);
	if(col == 1) {
        	fprintf(gp, "set yr [0:2.4]\n");
		colstr = "L";
	} else
	        fprintf(gp, "set autoscale\n");

	if(sample_loc == -1) {
        	fprintf(gp, "plot \"<(sed -n '1,%dp' %s)\" using %d title '%s' %s\n", sample_size, fname, col, colstr, type);
		if(col == 1) 
        		fprintf(gp, "\"<(sed -n '1,%dp' %s)\" using ($2 * 2) title 'R'\n", sample_size, fname);
	} else if (sample_loc == 0) {
        	fprintf(gp, "plot \"<(sed -n '%d,180000p' %s)\" using %d title '%s' %s\n", lastlines, fname, col, colstr, type);
		if(col == 1) 
        		fprintf(gp, "\"<(sed -n '%d,180000p' %s)\" using ($2 * 2) title 'R'\n", lastlines, fname);
	} else {
        	fprintf(gp, "plot \"%s\" every %d using %d title '%s' %s\n", fname, sample_period, col, colstr, type);
		if(col == 1) 
        		fprintf(gp, "\"%s\" every %d using ($2 * 2) title 'R'\n", fname, sample_period);
	}
	if(col != 2) printf("%s created\n", output);
}

int main(int argc, char **argv)
{
        gp = popen(GNUPLOT,"w"); /* 'gp' is the pipe descriptor */
        if (gp==NULL)
           {
             printf("Error opening pipe to GNU plot. Check if you have it! \n");
             exit(0);
           }
 
        fprintf(gp, "set terminal png\n");
	printf("source file: %s\n", fname);

	// left, right, r_hat[0], r_hat[1], theta, theta_dot, h, h_dot, force 
	// 0 0 0.0113 0.0113 -0.0755 -0.0499 0.9063 0.6871 0.0000

	// L/R for first, last, sampled steps
	plot(-1, 1);
	plot(0, 1);
	plot(1, 1);

	// rhat_L, rhat_R
	plot(-1, 3);
	plot(0, 3);
	plot(1, 3);

	plot(-1, 4);
	plot(0, 4);
	plot(1, 4);

	// force for first, last, sampled steps
	plot(-1, 9);
	plot(0, 9);
	plot(1, 9);

	// theta and thetadot for sampled steps
	plot(-1, 5);
	plot(0, 5);
	plot(1, 5);

	plot(-1, 6);
	plot(0, 6);
	plot(1, 6);

	// h and hdot for sampled steps
	plot(-1, 7);
	plot(0, 7);
	plot(1, 7);

	plot(-1, 8);
	plot(0, 8);
	plot(1, 8);
	
        fclose(gp);

	return 0;
}
