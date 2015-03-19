/* 
   v0.7.6 - 3/16/2015 @author Tae Seung Kang
   Spike Response Model (SRM) with continuous force version

   Changelog
   - output only SRM: p[i] 
   - Q*10: weight increased
   - action network of SRM)neurons: at all layers (input, hidden, output)
   - encode the states into input spikes: normalize to [0, 1] with threshold 0.5
   - lookup table to reduce computation time: <time, force> or <time, membrane potential>  
   - force computation time reduced: 30-50%
   - change sync error: rhat+=0.1 to rhat-=0.01
   - print the best results: cp latest.test best.test
   - mutex for sparse asnychronous fires 
   - sync error added to backprop
   - suppress flag: if fired last time, don't fire. suppress 1 spike
   - bug found: pushes[step] was missing in integrating force. not working after fix
     added pushes[200] to store up to the last 200 push values
   - total elapsed time while running
   - add spike error function to remove redundant spikes
   - changed the input arguments to take fm, dt, tau, and last_steps
   - higher force: 10 -> 50 same as cont force. 50 is the best
   - report stats: firing rates (L/R), rhats (L/R), state (4), force
     writes the data to a file latest.dat
   - 1output with 2actions(L/R) to 2outputs(L/R) with 3actions L/R/0
   - two outputs for both networks
   - td-backprop code for evaluation network combined: multiple outputs

   Todo list
   - xrates: firing rates for x
   - omp parallel for
   - different Q values: Q*5
   - double register for speedup: effective? size matters
   - in-memory printout: hold output in memory until last file write
   - just allocate large memory for last spikes? last_spike_p[i][3600000]: slow due to too many spikes?
   - optimization for speedup: too slow now => due to high rate?
   - estimated remaining time
   - rollout: 10k, 50k, 100k, 150k, 180k milestones or midpoints
   - recurrent outputs to affect each other: inhibit weights
   - test log files: 180k-fm50-r1.test1 .. test100, r1.train, r1.log, r1.weights
   - config file

   Discussion
   - is cyclone the fastest? others slow due to small cache (512k vs 6MB)
	 (and possibly file IO - cyclone is nfs server)
   - large variation in firing rates for the given max force fm
*/
/*********************************************************************************
    This file contains a simulation of the cart and pole dynamic system and 
 a procedure for learning to balance the pole that uses multilayer connectionist 
 networks.  Both are described in Anderson, "Strategy Learning with Multilayer
 Connectionist Representations," GTE Laboratories TR87-509.3, 1988.  This is
 a corrected version of the report published in the Proceedings of the Fourth
 International Workshop on Machine Learning, Irvine, CA, 1987.  

    main takes six arguments: 
        the maximum number of trials (balancing attempts),
        the number of trials over which balancing steps are averaged and printed
        beta
        beta for hidden units
        rho
        rho for hidden units

    Please send questions and comments to anderson@cs.colostate.edu
*********************************************************************************/
#include <stdio.h>
#include <math.h>
#include <sys/types.h>
#include <time.h>
#include <stdlib.h>

#define SRM
//#define PRINT_FAILURE
//#define SYNERR		0.001
//#define PRINT		  // print out the results
#define Q		0.001 // PSP synaptic weight amplifier
//#define Q		1.0 // PSP synaptic weight amplifier
//#define IMPULSE	
#define randomdef       ((float) random() / (float)((1 << 31) - 1))

/* SRM constants */
// PSP: AMPA for excitatory and GABAA for inhibitory
// AMPA (beta 1.0, tau_exc 0.02, tau_inh 0.01)
// NMDA (beta 5.0, tau 0.08)
// GABAA (beta 1.1, tau-exc 0.02, tau-inh 0.01)
// GABAB (beta 50, tau 0.1)
#define	beta		1.0
#define dist		1.5	// [1.0, 2.0] 1.5 for excitatory, 1.0-1.2 for inhibitory
#define tau_exc		20	// ms
// AHP
#define R		-1.0	// for AHP
//#define R		-1000	// originally 1000 for AHP
#define gamma		1.2	// 1.2 msec. for AHP

/* cart pole constants */
#define Mc           1.0 	// cart mass
#define Mp           0.1	// pole mass
#define l            0.5	// pole half length
#define g            9.8	
#define max_cart_pos 2.4
#define max_cart_vel 1.5
#define max_pole_pos 0.2094
#define max_pole_vel 2.01

#define Gamma 	     0.9  /* discount-rate parameter (typically 0.9) */
float Beta =  0.2;	/* 1st layer learning rate (typically 1/n) */
float Beta_h = 0.05;	/* 2nd layer learning rate (typically 1/num_hidden) */
float Rho = 1.0;	/* 1st layer learning rate (typically 1/n) */
float Rho_h = 0.2;	/* 2nd layer learning rate (typically 1/num_hidden) */
float LR_IH = 0.7;
float LR_HO = 0.07;
float state[4] = {0.0, 0.0, 0.0, 0.0};

//float Q = 1.0;	// [-10, 10]. connection strength randomly chosen from [1.0, 10.0]

struct
{
  double           cart_pos;
  double           cart_vel;
  double           pole_pos;
  double           pole_vel;
} the_system_state;

int start_state, failure;
double a[5][5], b[5][2], c[5][2], d[5][5], e[5][2], f[5][2]; 
double x[5], x_old[5], y[5], y_old[5], v[2], v_old[2], z[5], p[2];
double r_hat[2], push, unusualness[2], fired[2], pushes[3600000], forceValues[100];
int last_spike_p[2][100], last_spike_x[5][100], last_spike_v[5][100], last_spike_z[5][100];
//int xspikes[5], yspikes[5], zspikes[5], vspikes[5];
//int xspikes_old[5], yspikes_old[5], zspikes_old[5], vspikes_old[5];
double PSPValues[100], AHPValues[20];
double threshold = 0.03, pmin = 1000, pmax = -1000, zmin = 1000, zmax = -1000;

/* experimental parameters */
float fm = 50; 		// magnitude of force. 50 best, 25-100 good, 10 too slow
float dt = 0.001;	// 1ms step size
float tau = 0.02; 	// 20ms time constant
int DEBUG = 0;
int TEST_RUNS = 10;
int TARGET_STEPS = 5000;
int last_steps = 100, max_steps = 0; // global max steps so far
int balanced = 0, rspikes, lspikes, mutex = -1;
int test_flag = 0;
time_t gstart; // global timer
char *datafilename = "latest.train"; // latest.test1
char *best = "best.train", ch;
FILE *datafile, *bestfile;

/*** Prototypes ***/
float scale (float v, float vmin, float vmax, int devmin, int devmax);
void init_args(int argc, char *argv[]);
void eval();
void action(int step);
void updateweights();
void readweights(char *filename);
void writeweights();
float sign(float x) { return (x < 0) ? -1. : 1.;}

/* SRM */
double srm(int time, double weight);
//typedef enum {FORCE, PSP, AHP} strategy_t;
//strategy_t my_strategy = IMMEDIATE;
//double lookup(char *type, double t);
//double put(char *type, double t, double value);

void init_last_spikes() {
  int i, j;
  for(i = 0;i < 100; i++) {
    last_spike_p[0][i] = -1;
    last_spike_p[1][i] = -1;
    for(j = 0; j < 5; j++) {
      last_spike_x[j][i] = -1;
      last_spike_v[j][i] = -1;
      last_spike_z[j][i] = -1;
    }
  }
}

void init_constant_values() {
  int i, j;
  for(i = 0;i < last_steps; i++) {
    forceValues[i] = -1;
    PSPValues[i] = -1;
    if(i >= 20) continue;
    AHPValues[i] = 1;
  }
  init_last_spikes();
}

main(argc,argv)
     int argc;
     char *argv[];
{
  char a;

  setbuf(stdout, NULL);

  // [graphic] [target_steps] [test_runs] [fm] [dt] [tau] [last_steps] [debug] [max_trial] [sample_period] [weights]
  //  1		2		3	4    5     6	7		8    9			10 	   11
  init_args(argc,argv);
  init_constant_values();

  printf("balanced %d test_flag %d\n", balanced, test_flag);

  int i = 0;
  int num_trials = atoi(argv[9]);
  int sample_period = atoi(argv[10]);
  time_t start, stop, istart, istop;
  //tic
  time(&gstart);
  if(test_flag) {
    int trials, sumTrials = 0, maxTrials = 1, minTrials = 100, success = 0, maxSteps = 0;
    printf("TEST_RUNS = %d\n", TEST_RUNS);
    while(!balanced && i < 100) {
      printf("[Test Run %d] ", ++i);
      datafilename = "latest.test"; best = "best.test";
      trials = Run(num_trials, sample_period); // max_trial, sample_period
      sumTrials += trials;
      if(trials > maxTrials) maxTrials = trials;
      if(trials < minTrials) minTrials = trials;
      if(trials <= 100) success ++;
      if(balanced) break;
      init_args(argc,argv);
    }
    printf("\n=============== SUMMARY ===============\n");
    printf("Trials: %.2f\% (%d/%d) max %d steps\n", 
	100.0*success/TEST_RUNS, success, TEST_RUNS, max_steps);
  } else { 
    while(!balanced && i < 1000) {
      printf("[%d] ", ++i);
      Run(num_trials, sample_period);
      init_args(argc,argv);
    }
  }

  // toc
  time(&stop);
  printf("Total Elapsed time: %.0f seconds\n", difftime(stop, gstart));

  fclose(datafile);
  printf("Wrote current data to %s\n",datafilename);  
}

/**********************************************************************
 *
 **********************************************************************/
void init_args(int argc, char *argv[])
{
  int runtimes;
  struct timeval current;

  fired[0] = -1; fired[1] = -1; //mutex = -1;
  gettimeofday(&current, NULL);
  srandom(current.tv_usec);
  // [graphic] [target_steps] [test_runs] [fm] [dt] [tau] [last_steps] [debug] [max_trial] [sample_period] [weights]
  //  1		2		3	4    5     6	7		8    9			10 	   11
  if (argc < 5)
    exit(-1);
  if (argc > 2)
    TARGET_STEPS = atoi(argv[2]);
  if (argc > 3)
    TEST_RUNS = atoi(argv[3]);
  if (argc > 6) 
    tau = atof(argv[6]);
  fm = atof(argv[4]); 
  dt = atof(argv[5]);
  if (argc > 7)
    DEBUG = atoi(argv[8]);
  last_steps = atoi(argv[7]);
  if (strcmp(argv[11],"-") != 0) {
    readweights(argv[11]); test_flag = 1;
  } else {
    SetRandomWeights(); test_flag = 0;
  }
  //printf("[graphic] [target_steps] [test_runs] [fm] [dt] [tau] [debug] [max_trial] [sample_period] [weights]\n");
  //printf("%s %d %d %f %f %f %d %d %d %s\n", argv[1], TARGET_STEPS, TEST_RUNS, fm, dt, tau, DEBUG, atoi(argv[8]), atoi(argv[9]), argv[10]);
}

SetRandomWeights()
{
  int i,j;

  for(i = 0; i < 5; i++)
    {
      for(j = 0; j < 5; j++)
	{
	  a[i][j] = randomdef * 0.2 - 0.1;
	  d[i][j] = randomdef * 0.2 - 0.1;
	}
      for(j = 0; j < 2; j++) {
        b[i][j] = randomdef * 0.2 - 0.1;
        c[i][j] = randomdef * 0.2 - 0.1;
        e[i][j] = randomdef * 0.2 - 0.1;
        f[i][j] = randomdef * 0.2 - 0.1;
      }
    }
}

/****************************************************************/
/* If init_flag is zero, then calculate state of cart-pole system at time t+1
   by Euler's method, else set state of cart-pole system to random values.
*/
NextState(int init_flag, double push, int step)
     //int init_flag;
     //double push;
{
  register double pv, ca, pp, pa, common;
  double sin_pp, cos_pp;
  
  if (init_flag)
    {
      the_system_state.cart_pos = randomdef * 2 * max_cart_pos - max_cart_pos;
      the_system_state.cart_vel = randomdef * 2 * max_cart_vel - max_cart_vel;
      the_system_state.pole_pos = randomdef * 2 * max_pole_pos - max_pole_pos;
      the_system_state.pole_vel = randomdef * 2 * max_pole_vel - max_pole_vel;

      start_state = 1;
      SetInputValues(step);
      failure = 0;
    }
  else
    {
      pv = the_system_state.pole_vel;
      pp = the_system_state.pole_pos;
  
      sin_pp = sin(pp);
      cos_pp = cos(pp);

      common = (push + Mp * l * pv * pv * sin_pp) / (Mc + Mp);
      pa = (g * sin_pp - cos_pp * common) /
        (l * (4.0 / 3.0 - Mp * cos_pp * cos_pp / (Mc + Mp)));
      ca = common - Mp * l * pa * cos_pp / (Mc + Mp);
  
      the_system_state.cart_pos += dt * the_system_state.cart_vel;
      the_system_state.cart_vel += dt * ca;
      the_system_state.pole_pos += dt * the_system_state.pole_vel;
      the_system_state.pole_vel += dt * pa;

      SetInputValues(step);

      start_state = 0;
      if ((fabs(the_system_state.cart_pos) > max_cart_pos) ||
	  (fabs(the_system_state.pole_pos) > max_pole_pos))
	failure = -1;
      else
	failure = 0;
    }
}

/****************************************************************/
encode(double x) {
  float tau_encode = 0.1;
  return sin(x) * exp(-x/tau_encode);
}

// Normalize to [0, 1] and encode to spikes
SetInputValues(int step)
{
  x[0] = (the_system_state.cart_pos + max_cart_pos) / (2 * max_cart_pos);
  x[1] = (the_system_state.cart_vel + max_cart_vel) / (2 * max_cart_vel);
  x[2] = (the_system_state.pole_pos + max_pole_pos) / (2 * max_pole_pos);
  x[3] = (the_system_state.pole_vel + max_pole_vel) / (2 * max_pole_vel);
  x[4] = 0.5;
  int i;
  for(i = 0; i < 5; i ++) {
    //x[i] = encode(x[i]);
    //if(x[i] >= 0.3) {
    if(x[i] >= 0.5) {
	last_spike_x[i][step%100] = step;
	//printf("  x[%d] fires at step %d slot %d\n", i, step, step%200);
    } else
	last_spike_x[i][step%100] = -1;
/*
    if(step%100 == 0) { 
      int j, xcount = 0; 
      double xrate;
      for(j = 0; j < 100; j++) {
	if(last_spike_x[i][j] != -1) xcount ++;
	//xrate = (xcount - xoldcount)/100;
      }
      xrate = xcount / (dt*100.0);
      printf("xrate %f /sec, %f /setp (%d)\n", xrate, xcount / 100.0, xcount);
    }
*/  }
}

/****************************************************************/
// returns the number of trials before failure
int Run(num_trials, sample_period)
 int num_trials, sample_period;
{
  // maxj: among current trial
  // max_length: among current run
  // max_steps: among whole runs
  register int i, j, avg_length, max_length = 0, maxj, maxlspk, maxrspk;
  time_t start, stop; 
  lspikes = 0; rspikes = 0; //mutex = -1;

  time(&start);

  NextState(1, 0.0, 0);
  i = 0;   j = 0;
  avg_length = 0;
  init_last_spikes();

    if ((datafile = fopen(datafilename,"w")) == NULL) {
      printf("Couldn't open %s\n",datafilename);
      return;
    }

  while (i < num_trials && j < TARGET_STEPS) /* one hour at .02s per step */
    {
      Cycle(1, j, sample_period);
    //  if (DEBUG && j % 1000 == 0)
    //    printf("Episode %d step %d rhat %.4f\n", i, j, r_hat);
      j++;

      if (failure)
	{
   	  //max_length = (max_length < j ? j : max_length);
#ifdef PRINT_FAILURE
	    printf("\t%d step %d max %d rate %f (L%d:R%d) %f %f %f %f\n", i, j, max_length, 
		(lspikes + rspikes)/(dt*j), lspikes, rspikes, pmin, pmax, zmin, zmax);
#endif
	  if(maxj < j) {
	    maxj = j; 
	    maxlspk = lspikes; maxrspk = rspikes;
	    if(max_length < j) {
	  	max_length = j;
 	    }
 	  }
#ifdef PRINT
  if(max_steps < j) {
    if ((bestfile = fopen(best,"w")) == NULL) {
      printf("Couldn't open %s\n",best);
      return;
    }
    // copy latest.train to best.train
    while((ch = fgetc(datafile)) != EOF)
	fputc(ch, bestfile);
    fclose(bestfile);
  }
#endif
	  j = 0; lspikes = 0; rspikes = 0; //mutex = -1;
  	  fclose(datafile);
     	  if ((datafile = fopen(datafilename,"w")) == NULL) {
      	    printf("Couldn't open %s\n",datafilename);
      	    return;
          }
  	  init_last_spikes();
	  i++;
	  NextState(1, 0.0, 0);
	}
    } // end while loop for single trial

   if(i >= num_trials) {
     balanced = 0;
     max_steps = (max_steps < max_length ? max_length : max_steps);
     printf("Max %d (%d) steps (%.4f hrs) ",
            max_steps, max_length, (max_length * dt)/3600.0);
   } else {
     printf("Ep%d balanced for %d steps (%.4f hrs). ",
            i, j, (j * dt)/3600.0);
     balanced = 1;
   }

   time(&stop);
   double tt = maxj*dt; // total time
   printf("%.2f:%.2f %.0f (%.0f) sec\n", maxlspk/tt, maxrspk/tt, difftime(stop, gstart), difftime(stop, start));

  if(balanced) 
  {
    if(!test_flag) writeweights();

    tt = j*dt;
    fprintf(datafile,"\n%.2f spikes/sec (L:%.2f R:%.2f)\n", (lspikes + rspikes)/(tt), lspikes/(tt), rspikes/(tt));
    fprintf(datafile,"%.2f spikes/step (L:%.2f R:%.2f)\n", ((double)(lspikes + rspikes))/(double)j, lspikes/(double)j, rspikes/(double)j);
    fprintf(datafile,"%d spikes (L:%d R:%d), j = %d, dt = %.4f\n", (lspikes + rspikes), lspikes, rspikes, j, dt);
  }

  return i + 1;
}

/****************************************************************/

double sgn(x)
     double x;
{
  if (x < 0.0)
    return -1.0;
  else if (x > 0.0)
    return 1.0;
  else
    return 0.0;
}

double getForce(int step) {
  double force = forceValues[step];
  if(force == -1) {
    double t = dt*step;
    force = t * exp(-t/tau);
    forceValues[step] = force;
  }
  return force;
}

/****************************************************************/

Cycle(learn_flag, step, sample_period)
     int learn_flag, step, sample_period;
{
  int i, j, k, left = 0, right = 0;
  double sum, factor1, factor2, t;
  extern double exp();
  float state[4];

  /* output: state evaluation */
  eval();

  /* output: action */
  action(step);

  if(randomdef <= p[0]) {
  //if(p[0] >= 0.004) {
//    printf("  p[0] fires %d slot %d: ", step, step%100);
    //p[0] = 1.0 / (1.0 + exp(-sum));
    last_spike_p[0][step%100] = step;
/*
    for(k = 0; k < 100; k++) 
	printf("%d ", last_spike_p[0][k]);
    printf("\n");
*/
    left = 1; lspikes ++;
    unusualness[0] = 1 - p[0];
  } else {
    unusualness[0] = -p[0];
    last_spike_p[0][step%100] = -1;
  }

  if(randomdef <= p[1]) { 
  //if(p[1] >= 0.004) {
  //  printf("  p[1] fires %d slot %d: ", step, step%100);
    last_spike_p[1][step%100] = step;
/*
    for(k = 0; k < 100; k++) 
	printf("%d ", last_spike_p[1][k]);
    printf("\n");
*/
    right = 1; rspikes ++;
    unusualness[1] = 1 - p[1];
  } else {
    unusualness[1] = -p[1];
    last_spike_p[1][step%100] = -1;
  }

  if(left == 1 && right == 0) {
    push = 1.0; 
  } else if (left == 0 && right == 1) {
    push = -1.0; 
  } else  
    push = 0; 

#ifdef IMPULSE
  push *= fm;
#else
  pushes[step] = push; // problematic in accessing index step
  sum = 0.0;
  int upto = (step > last_steps ? last_steps: step);
  for(i = 1; i < upto ; i++) {
    //t = i * dt;
    sum += pushes[step - i] * getForce(i);
    //sum += pushes[step - i] * t * exp(-t/tau);
  }
  push = fm*sum;
#endif

  /* preserve current activities in evaluation network. */
  for (i = 0; i< 2; i++)
    v_old[i] = v[i];

  for (i = 0; i< 5; i++)
  {
    x_old[i] = x[i];
    y_old[i] = y[i];
  }

  /* Apply the push to the pole-cart */
  NextState(0, push, step);

  /* Calculate evaluation of new state. */
  eval();

  /* action evaluation */
  for(i = 0; i < 2; i++) {
    if (start_state)
      r_hat[i] = 0.0;
    else {
      if (failure) {
        r_hat[i] = failure - v_old[i];
     } else {
        r_hat[i] = failure + Gamma * v[i] - v_old[i];
     }
#ifdef SYNERR
     if(left == 1 && right == 1)
        r_hat[i] -= SYNERR;
#endif
     }
  }
  /* report stats */
#ifdef PRINT
//  if(step % sample_period == 0)
    fprintf(datafile,"%d %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", left, right, r_hat[0], r_hat[1], 
			the_system_state.pole_pos, the_system_state.pole_vel, 
			the_system_state.cart_pos, the_system_state.cart_vel,
 			push);
#endif
  /* modification */
  if (learn_flag)
	updateweights();
}

/**********************************************************************/
// lookup table from step 0 to 99 to speed up computation
double PSP(int step) {
  if(step == 0) return 0;
  double psp = PSPValues[step];
  if(psp == -1) {
    //double t = dt * step;
    double t = 1000 * dt * step;
    //t = dt*(step - last_spike_z[i][k]);
    psp = (1.0/dist*sqrt(t)) * exp(-beta*dist*dist/t) * exp(-t/tau_exc);
    PSPValues[step] = psp;
/*
    printf("  step %d psp %f t %f (exp %lf %lf %f): ", step, psp, t, exp(-beta*dist*dist/t), -beta*dist*dist/t, dist*dist);
    int i = 0;
    for(;i < 100; i ++)
	printf("%.2f ", PSPValues[i]);
     printf("\n");
*/  }
  //sum += e[i][j]*10.0/(dist*sqrt(t)) * exp(-beta*dist*dist/t) * exp(-t/tau_exc);
//  usleep(1000);
  return psp;
}

double AHP(int step) {
  if(step >= 20) return 0;
  double ahp = AHPValues[step];
  if(ahp == 1) {
    //double t = dt * step;
    double t = 1000 * dt * step;
    ahp = R * exp(-t/gamma);
    AHPValues[step] = ahp;
/*
    printf("  step %d ahp %f t %f: ", step, ahp, t);
    int i = 0;
    for(;i < 20; i ++)
	printf("%.2f ", AHPValues[i]);
     printf("\n");
*/
  }
  return ahp;
  //return R * exp(-t/gamma);
}

/**********************************************************************/
void eval() {
  int i, j;
  double sum;
  for(i = 0; i < 5; i++)
    {
      sum = 0.0;
      for(j = 0; j < 5; j++)
	  sum += a[i][j] * x[j];
      y[i] = 1.0 / (1.0 + exp(-sum));
    }
  for (j = 0; j< 2; j++) {
    sum = 0.0;
    for(i = 0; i < 5; i++)
      sum += b[i][j] * x[i] + c[i][j] * y[i];
    v[j] = sum;
  }
}

void action(int step) {
  //printf("  action step %d\n", step);
  int i, j, k;
  double sum, t, tk, psp, ahp;
  for(i = 0; i < 5; i++)
    {
      sum = 0.0;
      for (j = 0; j < 5; j++) {
	sum += d[i][j] * x[j];
      //z[i] = 1.0 / (1.0 + exp(-sum));
//	printf("d[j][i] %f ", d[j][i]);
/*
        for(k = 1; k < 100; k ++) {
  	  if(last_spike_x[j][k] != -1) {
	    int _steps = step - last_spike_x[j][k];
	    psp = PSP(step - last_spike_x[j][k]);
	    //psp = d[j][i] * Q * PSP(step - last_spike_x[j][k]);
	    //sum += d[j][i] * Q * PSP(step - last_spike_x[j][k]);
	    sum += d[j][i] * Q * psp;
	    if(sum > 1000 || sum < -1000 || psp > 5 || psp < -5 || isnan(d[j][i]) || isinf(d[j][i]) || isnan(psp) || isinf(psp)) {
		printf("step %d PSPs %f PSP %f timesteps %d d[j][i] %f\n", step, sum, psp, _steps, d[j][i]);
		exit(1);
	    }
//	    printf("%f ", psp);
	  }
	}
      }
      psp = sum;
 //     printf("PSPs %f ", sum);
      ahp = 0;
      for(k = 0; k < 20; k ++) {
  	if(last_spike_z[i][k] != -1) {
	  ahp += AHP(step - last_spike_z[i][k]);
	  //sum += AHP(step - last_spike_z[i][k]);
	}
*/     
      }
      //sum += ahp;

      z[i] = 1.0 / (1.0 + exp(-sum));
      //z[i] = sum/1000.0;
      //z[i] = sum/10.0;
/*
      if(z[i] < zmin) zmin = z[i];
      if(z[i] > zmax) zmax = z[i];
      if(z[i] > 1.0 || z[i] < -1.0 || isnan(z[i]) || isinf(z[i])) {
        printf("z[%d] = %f PSPs %f AHPs %f d[j][i] %f\n", i, z[i], psp, ahp, d[j][i]);
	printf("AHPValues: ");
	int _i;
    	for(_i = 0;_i < 20; _i ++)
	  printf("%.2f ", AHPValues[_i]);
        printf("\n");
      	for(k = 0; k < 20; k ++) {
  	  if(last_spike_z[i][k] != -1) {
	    int _steps = step - last_spike_z[i][k];
	    ahp = AHP(_steps);
	    printf("ahp %f time %d step %d last_spike_z[i][k] %d\n", ahp, _steps, step, last_spike_z[i][k]);
	    //sum += AHP(step - last_spike_z[i][k]);
	  }
  	}
	printf("PSPValues: ");
    	for(_i = 0;_i < 100; _i ++)
	  printf("%.2f ", PSPValues[_i]);
        printf("\n");
	printf("last_spike_z[i][k]: ");
        for(k = 0; k < 100; k++) 
	  printf("%d ", last_spike_z[i][k]);
        printf("\n");
        //printf("\n");
	exit(1);
      }
*/      //usleep(1000);
      if (z[i] >= randomdef) {
	last_spike_z[i][step%100] = step;
/*
	printf("  z[%d] fires at step %d slot %d: ", i, step, step%100);
        for(k = 0; k < 100; k++) 
	  printf("%d ", last_spike_z[i][k]);
        printf("\n");
*/
      }
      else last_spike_z[i][step%100] = -1;
    }
  for (j = 0; j < 2; j++) {
    sum = 0.0;
    // for PSPs
    for(i = 0; i < 5; i++) {
	// last spikes of neuron i at x and z
	for(k = 1; k < 100; k ++) {
	  if(last_spike_x[i][k] != -1)
	    sum += e[i][j]* Q *PSP(step - last_spike_x[i][k]);
	  if(last_spike_z[i][k] != -1) {
//	    printf("last_spike_z %d %d %d psp %f\n", step, i, k, psp);
	  //sum += Q/(dist*sqrt(t)) * exp(-beta*dist*dist/t) * exp(-t/tau_exc);
	  //sum += e[i][j]*10.0/(dist*sqrt(t)) * exp(-beta*dist*dist/t) * exp(-t/tau_exc);
	    sum += f[i][j]* Q *PSP(step - last_spike_z[i][k]);
	  }
	}
    }
    psp = sum;
    //p[j] = sum + R * exp(-t/gamma);
    for(k = 0; k < 20; k ++) 
      if(last_spike_p[j][k] != -1) 
        sum += AHP(step - last_spike_p[j][k]);
    //p[j] = sum/10.0; // break, not working
    //p[j] = sum;
    //p[j] = sum / 50.0;
    //p[j] = sum / 100.0; // too small
      //sum += e[i][j] * x[i] + f[i][j] * z[i];
    p[j] = 1.0 / (1.0 + exp(-sum));
/*
    if(p[j] < pmin) pmin = p[j];
    if(p[j] > pmax) pmax = p[j];
    if(p[j] < -1.0 || p[j] > 1.0) {
	printf("  p[%d] %f PSPs %f AHPs %f\n", j, p[j], psp, sum - psp);
	break;
    }
*/
  }
  //printf("  p[j] %f\n", p[j]);
}

/**********************************************************************/
void updateweights() {
  int i, j, k;
  double factor1, factor2;
      for(i = 0; i < 5; i++)
	{
	  for(j = 0; j < 5; j++)
	    {
	      for(k = 0; k < 2; k++) {
   	        factor1 = Beta_h * r_hat[k] * y_old[i] * (1.0 - y_old[i]) * sgn(c[i]);
	        factor2 = Rho_h * r_hat[k] * z[i] * (1.0 - z[i]) * sgn(f[i]) * unusualness[k];
   	        a[i][j] += factor1 * x_old[j];
	        d[i][j] += factor2 * x_old[j];
		//if(d[i][j] != d[i][j]) 
		if(d[i][j] > 100 || d[i][j] < -100 || isnan(d[i][j])) {
		//if(isnan(d[i][j]) || isinf(d[i][j])) {
		  printf("d[i][j] %f x_old[j] %f r_hat[k] %f z[i] %f unusualness[k] %f p[0] %f p[1] %f\n",
			d[i][j], x_old[j], r_hat[k], z[i], unusualness[k], p[0], p[1]);
		  exit(1);
		}
              }
	    }
	  for(j = 0; j < 2; j++) {
	    b[i][j] += Beta * r_hat[j] * x_old[i];
	    c[i][j] += Beta * r_hat[j] * y_old[i];
	    e[i][j] += Rho * r_hat[j] * unusualness[j] * x_old[i];
	    f[i][j] += Rho * r_hat[j] * unusualness[j] * z[i];
  	  }
	}
    }

/**********************************************************************/

void readweights(filename)
char *filename;
{
  int i,j;
  FILE *file;

  if ((file = fopen(filename,"r")) == NULL) {
    printf("Couldn't open %s\n",filename);
      exit(1);
//    return;
  }

  for(i = 0; i < 5; i++)
      for(j = 0; j < 5; j++)
	  fscanf(file,"%lf",&a[i][j]);

  for(i = 0; i < 5; i++)
      for(j = 0; j < 2; j++)
        fscanf(file,"%lf",&b[i][j]);

  for(i = 0; i < 5; i++)
      for(j = 0; j < 2; j++)
        fscanf(file,"%lf",&c[i][j]);


  for(i = 0; i < 5; i++)
      for(j = 0; j < 5; j++)
	  fscanf(file,"%lf",&d[i][j]);

  for(i = 0; i < 5; i++)
      for(j = 0; j < 2; j++)
        fscanf(file,"%lf",&e[i][j]);

  for(i = 0; i < 5; i++)
      for(j = 0; j < 2; j++)
       fscanf(file,"%lf",&f[i][j]);

  fclose(file);

  if(!test_flag)
    printf("Read weights from %s\n",filename);
}


void writeweights()
{
  char *filename = "latest.weights";
  int i,j;
  FILE *file;

  if ((file = fopen(filename,"w")) == NULL) {
    printf("Couldn't open %s\n",filename);
    return;
  }

  for(i = 0; i < 5; i++)
      for(j = 0; j < 5; j++)
	  fprintf(file," %f",a[i][j]);

  fprintf(file, "\n");

  for(i = 0; i < 5; i++)
      for(j = 0; j < 2; j++)
       fprintf(file," %f",b[i][j]);

  fprintf(file, "\n");

  for(i = 0; i < 5; i++)
      for(j = 0; j < 2; j++)
        fprintf(file," %f",c[i][j]);

  fprintf(file, "\n");

  for(i = 0; i < 5; i++)
      for(j = 0; j < 5; j++)
	  fprintf(file," %f",d[i][j]);

  fprintf(file, "\n");

  for(i = 0; i < 5; i++)
      for(j = 0; j < 2; j++)
        fprintf(file," %f",e[i][j]);

  fprintf(file, "\n");

  for(i = 0; i < 5; i++)
      for(j = 0; j < 2; j++)
       fprintf(file," %f",f[i][j]);

  fclose(file);

  printf("Wrote current weights to %s\n",filename);
}
