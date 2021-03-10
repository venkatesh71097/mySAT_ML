#include <errno.h>

#include <signal.h>
#include <zlib.h>
#include <sys/resource.h>

#include "utils/System.h"
#include "utils/ParseUtils.h"
#include "utils/Options.h"
#include "core/Dimacs.h"
#include "core/Solver.h"

using namespace Minisat;

//=================================================================================================


void printStats(Solver& solver)
{
    //long double tot_conf_count = 0;
    //long double tot_decisions = 0;
    //double cpu_time = cpuTime();
    //double mem_used = memUsedPeak();
    /*printf("OK\n");
    //printf("restarts              : %"PRIu64"\n", solver.starts);
    //printf("conflicts             : %-12"PRIu64"   (%.0f /sec)\n", solver.conflicts   , solver.conflicts   /cpu_time);
    printf("decisions             : %-12"PRIu64"   (%4.4f %% random) (%.0f /sec)\n", solver.decisions, (float)solver.rnd_decisions*100 / (float)solver.decisions, solver.decisions   /cpu_time);
    //printf("propagations          : %-12"PRIu64"   (%.0f /sec)\n", solver.propagations, solver.propagations/cpu_time);
    //printf("conflict literals     : %-12"PRIu64"   (%4.2f %% deleted)\n", solver.tot_literals, (solver.max_literals - solver.tot_literals)*100 / (double)solver.max_literals);
    printf("CPU time              : %g s\n", cpu_time);
    //printf("conflict_analysis     : %-12"PRIu64"   (%.0f /sec)\n", solver.conflict_analysis, solver.conflict_analysis/cpu_time);
    //printf("learnt_conflict_analysis : %-12"PRIu64"   (%.0f /sec)\n", solver.learnt_conflict_analysis, solver.learnt_conflict_analysis/cpu_time);
#if br_heuristic == greedy_vsids
    printf("learnt_propagations   : %-12"PRIu64"   (%.0f /sec)\n", solver.learnt_propagations, solver.learnt_propagations/cpu_time);
    printf("It is Greedy VSIDS\n");

#endif
#if br_heuristic == sgd
    printf("conflict class size          : %-12"PRIu64"\n", solver.tot_conflicts);
    printf("nonconflict class size             : %-12"PRIu64"\n", solver.nonconflicts);
    printf("restarts              : %"PRIu64"\n", solver.starts);
    printf("It is SGD\n");
#endif
#if lbd_clause_del == 1
    printf("lbds                  : %-12"PRIu64"\n", solver.tot_lbds);
#endif
    for (int i = 0; i < solver.nVars(); i++) {
        tot_conf_count += solver.tot_conf_count[i]; //#conflicts
        tot_decisions += solver.tot_decisions[i]; //#decisions
    }
    printf("Global LR            : %Lf\n", tot_conf_count / tot_decisions);
    if (mem_used != 0) printf("Memory used           : %.4f MB\n", mem_used);*/
}


static Solver* solver;
// Terminate by notifying the solver and back out gracefully. This is mainly to have a test-case
// for this feature of the Solver as it may take longer than an immediate call to '_exit()'.
static void SIGINT_interrupt(int signum) { solver->interrupt(); }

// Note that '_exit()' rather than 'exit()' has to be used. The reason is that 'exit()' calls
// destructors and may cause deadlocks if a malloc/free function happens to be running (these
// functions are guarded by locks for multithreaded use).
static void SIGINT_exit(int signum) {
    printf("\n"); printf("*** INTERRUPTED ***\n");
    if (solver->verbosity > 0){
        printStats(*solver);
        printf("\n"); printf("*** INTERRUPTED ***\n"); }
    _exit(1); }


//=================================================================================================
// Main:

int main(int argc, char** argv)
{
    try {
        setUsageHelp("USAGE: %s [options] <input-file> <result-output-file>\n\n  where input may be either in plain or gzipped DIMACS.\n");
        // printf("This is MiniSat 2.0 beta\n");
        
#if defined(__linux__)
        fpu_control_t oldcw, newcw;
        _FPU_GETCW(oldcw); newcw = (oldcw & ~_FPU_EXTENDED) | _FPU_DOUBLE; _FPU_SETCW(newcw);
#endif
        // Extra options:
        //
        IntOption    verb   ("MAIN", "verb",   "Verbosity level (0=silent, 1=some, 2=more).", 1, IntRange(0, 2));
        BoolOption   pre    ("MAIN", "pre",    "Completely turn on/off any preprocessing.", true);
        StringOption dimacs ("MAIN", "dimacs", "If given, stop after preprocessing and write the result to this file.");
        StringOption assumptions ("MAIN", "assumptions", "If given, use the assumptions in the file.");
        IntOption    cpu_lim("MAIN", "cpu-lim","Limit on CPU time allowed in seconds.\n", INT32_MAX, IntRange(0, INT32_MAX));
        IntOption    mem_lim("MAIN", "mem-lim","Limit on memory usage in megabytes.\n", INT32_MAX, IntRange(0, INT32_MAX));

        parseOptions(argc, argv, true);
        
        Solver  S;
        double      initial_time = cpuTime();

        //if (!pre) S.eliminate(true);

        S.verbosity = verb;
        
        solver = &S;
        // Use signal handlers that forcibly quit until the solver will be able to respond to
        // interrupts:
        signal(SIGINT, SIGINT_exit);
        signal(SIGXCPU,SIGINT_exit);

        // Set limit on CPU-time:
        if (cpu_lim != INT32_MAX){
            rlimit rl;
            getrlimit(RLIMIT_CPU, &rl);
            if (rl.rlim_max == RLIM_INFINITY || (rlim_t)cpu_lim < rl.rlim_max){
                rl.rlim_cur = cpu_lim;
                if (setrlimit(RLIMIT_CPU, &rl) == -1)
                    printf("WARNING! Could not set resource limit: CPU-time.\n");
            } }

        // Set limit on virtual memory:
        if (mem_lim != INT32_MAX){
            rlim_t new_mem_lim = (rlim_t)mem_lim * 1024*1024;
            rlimit rl;
            getrlimit(RLIMIT_AS, &rl);
            if (rl.rlim_max == RLIM_INFINITY || new_mem_lim < rl.rlim_max){
                rl.rlim_cur = new_mem_lim;
                if (setrlimit(RLIMIT_AS, &rl) == -1)
                    printf("WARNING! Could not set resource limit: Virtual memory.\n");
            } }
        
        if (argc == 1)
            printf("Reading from standard input... Use '--help' for help.\n");

        gzFile in = (argc == 1) ? gzdopen(0, "rb") : gzopen(argv[1], "rb");
        if (in == NULL)
            printf("ERROR! Could not open file: %s\n", argc == 1 ? "<stdin>" : argv[1]), exit(1);
        
        
        parse_DIMACS(in, S);
        gzclose(in);
        FILE* res = (argc >= 3) ? fopen(argv[2], "wb") : NULL;
        double parsed_time = cpuTime();

        // Change to signal-handlers that will only notify the solver and allow it to terminate
        // voluntarily:
        signal(SIGINT, SIGINT_interrupt);
        signal(SIGXCPU,SIGINT_interrupt);

        //S.eliminate(true);
        double simplified_time = cpuTime();
        if (dimacs){
            S.toDimacs((const char*)dimacs);
            exit(0);
        }

        vec<Lit> dummy;
        if (assumptions) {
            const char* file_name = assumptions;
            FILE* assertion_file = fopen (file_name, "r");
            if (assertion_file == NULL)
                printf("ERROR! Could not open file: %s\n", file_name), exit(1);
            int i = 0;
            while (fscanf(assertion_file, "%d", &i) == 1) {
                Var v = abs(i) - 1;
                Lit l = i > 0 ? mkLit(v) : ~mkLit(v);
                dummy.push(l);
            }
            fclose(assertion_file);
        }
        for( int i = 0; i < dummy.size(); i++) {
            printf("%s%d\n", sign(dummy[i]) ? "-" : "", var(dummy[i]));
        }
        lbool ret = S.solveLimited(dummy);

        if (ret == l_True){
            printf("RESULT: SAT\nASSIGNMENT: ");
            for (int i = 0; i < S.nVars(); i++)
                if (S.model[i] != l_Undef)
                    printf("x%d=%s ", i+1, (S.model[i]==l_True)?"1":"0");
            //printf(" 0\n");
        }else if (ret == l_False) {
            printf("RESULT: UNSAT\n");
            for (int i = 0; i < S.conflict.size(); i++) {
                // Reverse the signs to keep the same sign as the assertion file.
                printf("%s%d\n", sign(S.conflict[i]) ? "" : "-", var(S.conflict[i]) + 1);
            }
        } else
            printf("INDET\n");
        //printf(ret == l_True ? "SATISFIABLE\n" : ret == l_False ? "UNSATISFIABLE\n" : "INDETERMINATE\n");
        if (res != NULL){
            if (ret == l_True){
                fprintf(res, "RESULT: SAT\nASSIGNMENT: ");
                for (int i = 0; i < S.nVars(); i++)
                    if (S.model[i] != l_Undef)
                        fprintf(res, "x%d=%s ", i+1, (S.model[i]==l_True)?"1":"0");
                //fprintf(res, " 0\n");
            }else if (ret == l_False) {
                fprintf(res, "RESULT: UNSAT\n");
                for (int i = 0; i < S.conflict.size(); i++) {
                    // Reverse the signs to keep the same sign as the assertion file.
                    fprintf(res, "%s%d\n", sign(S.conflict[i]) ? "" : "-", var(S.conflict[i]) + 1);
                }
            } else
                fprintf(res, "INDET\n");
            fclose(res);
        }

#ifdef NDEBUG
        exit(ret == l_True ? 10 : ret == l_False ? 20 : 0);     // (faster than "return", which will invoke the destructor for 'Solver')
#else
        return (ret == l_True ? 10 : ret == l_False ? 20 : 0);
#endif
    } catch (OutOfMemoryException&){
        printf("===============================================================================\n");
        printf("INDETERMINATE\n");
        exit(0);
    }
}
