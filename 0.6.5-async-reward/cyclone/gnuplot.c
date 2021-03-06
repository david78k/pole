/*
  v0.2 - 2/20/2015 @author Tae Seung Kang 

  Changelog
  - nsteps (number of steps) as input argument
  - multiplot

  Todo list
  -
*/
#include <stdio.h>
#include <stdlib.h>

#define GNUPLOT "gnuplot -persist"
//#define samplesize 2000 
//#define samplesize 1000 

int nsteps = 3600; // k (x1000)
int totalsteps = 10000;
//int totalsteps = 3600 * 1000;
char *prefix = "180k-test1"; // output file
//char *prefix = "180k-train";
char *fname = "180k-fm200-sup1-sample1-r1.test1"; // source file
int sample_period = 100; // 1, 10, 100 samples for plotting

int samplesize = 1000;  // first, last sample steps
int lastlines;
char output[30];
FILE *gp;

// sample_loc: first -1, last 0, all 1
//void plot(int sample_loc, int col) {
void plot(int col) {
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

	sprintf(output, "%s-%s.png", prefix, colstr);

	if(col != 2)
	        fprintf(gp, "set output '%s'\n", output);
	fprintf(gp, "set multiplot layout 3, 1\n");
	if(col == 1) {
        	fprintf(gp, "set yr [0:2.4]\n");
		colstr = "L";
	} else
	        fprintf(gp, "set autoscale\n");

	// all sampled
	//fprintf(gp, "set xtics (\"1800 (x100)\" 1800)\n", lastlines);
       	fprintf(gp, "set xr [0:%d]\n", totalsteps / sample_period);
	fprintf(gp, "set xlabel \"(x%d)\"\n", sample_period);
       	fprintf(gp, "plot \"%s\" every %d using %d title '%s' %s\n", fname, sample_period, col, colstr, type);
	if(col == 1) 
       		fprintf(gp, "\"%s\" every %d using ($2 * 2) title 'R'\n", fname, sample_period);
	// first steps
       	fprintf(gp, "set autoscale x\n");
	fprintf(gp, "unset xlabel\n");
        fprintf(gp, "plot \"<(sed -n '1,%dp' %s)\" using %d title '%s' %s\n", samplesize, fname, col, colstr, type);
	if(col == 1) 
        	fprintf(gp, "\"<(sed -n '1,%dp' %s)\" using ($2 * 2) title 'R'\n", samplesize, fname);
	// last steps
	//fprintf(gp, "set xtics %d,%d nomirror\n", lastlines, nsteps * 1000);
	int onestick = samplesize/5;
	fprintf(gp, "set xtics (\"%d\" 1, \"%d\" %d, \"%d\" %d, \"%d\" %d, \"%d\" %d, \"%d\" %d)\n", lastlines, 
				lastlines + onestick, (int)(0.2*samplesize), lastlines + 2*onestick, (int)(0.4*samplesize), 
				lastlines + 3*onestick, (int)(0.6*samplesize), totalsteps - onestick, (int)(0.8*samplesize), totalsteps, samplesize);
	//fprintf(gp, "set xtics (\"%d\" 1, \"%d\" 200, \"%d\" 400, \"%d\" 600, \"%d\" 800, \"%d\" 1000)\n", lastlines, lastlines + 200, 
	//			lastlines + 400, lastlines + 600, lastlines + 800, nsteps * 1000);
	//fprintf(gp, "set xtics (\"%d\" 1, \"%d\" 100, \"%d\" 200, \"%d\" 300, \"%d\" 400, \"%d\" 500)\n", lastlines, lastlines + 100, 
	//			lastlines + 200, lastlines + 300, lastlines + 400, nsteps * 1000);
        fprintf(gp, "plot \"<(sed -n '%d,%dp' %s)\" using %d title '%s' %s\n", lastlines, totalsteps, fname, col, colstr, type);
	if(col == 1) 
        	fprintf(gp, "\"<(sed -n '%d,%dp' %s)\" using ($2 * 2) title 'R'\n", lastlines, totalsteps, fname);
	if(col != 2) printf("%s created\n", output);
	fprintf(gp, "unset multiplot\n");
	fprintf(gp, "set xtic auto\n");
}

int main(int argc, char **argv)
{
	// [prefix] [source_file] [sample_period] [last_lines]
	// [train|test] [nsteps] [sample_period]
	prefix = argv[1];
	fname = argv[2];
	sample_period = atoi(argv[3]);
	lastlines = atoi(argv[4]);
	samplesize = atoi(argv[5]);
	nsteps = atoi(argv[6]);
	totalsteps = nsteps * 1000;

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
	plot(1);
	plot(3);
	plot(4);
	plot(9);
	plot(5);
	plot(6);
	plot(7);
	plot(8);

        fclose(gp);

	return 0;
}
