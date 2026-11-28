#!/bin/sh
# Run every measurement script in the order the paper's tables were produced.
# Each script writes its JSON (and, for the scripts that print a report, a log)
# next to itself; the copies in results/ and logs/ are the outputs quoted in the paper.
# Total runtime is about 70 minutes on one CPU core (pilot25 is about 40 of them and
# accepts --workers).  pilot11 writes
# pilot11_momentum.npz, which pilot14, pilot17 and pilot20 read; pilot20 also
# extends pilot19's grid and reads its JSON (../results/ in this layout).
set -e
cd "$(dirname "$0")/scripts"
for s in pilot1_polar_precision pilot1b_pareto pilot1c_precision_scaling \
         pilot2_bon_frontier pilot2b_frontier pilot3_bits_floor pilot3b_stability \
         pilot3c_rounding_schemes pilot4_precision_placement pilot4b_k1_separation \
         pilot4c_k1_highdeg pilot5_lower_bound pilot5b_scaling pilot5c_composition_design \
         pilot6_s_pert pilot6b_floor_constant pilot7_pert_aware_design pilot7b_c_derivation \
         pilot7c_table1 pilot8_muon_mlp pilot8b_equilibrium pilot9_candidate2_check \
         pilot10_subspace_theory pilot10b_first_step_schedule pilot11_momentum_subspace \
         pilot11b_application_order pilot12_arithmetic_variants pilot13_ell_sensitivity \
         pilot14_real_allocation pilot15_fp8_kernel pilot16_stats_and_lp \
         pilot17_theory_refinement pilot18_c_scaling_and_e5m2_fix pilot18b_bootstrap \
         pilot19_large_d_and_decomp pilot20_analytic_and_d1024 ; do
    echo "== $s"
    python3 "$s.py" > "$s_run.log" 2>&1 || { echo "   failed"; tail -5 "$s_run.log"; exit 1; }
done

# pilot21 is stage-driven, and the sequence below is the one that produced
# results/pilot21_results.json (the JSON keeps insertion order, so the stages
# appear in this order in the file).  Each stage resumes from the file, so a
# fresh directory is required to reproduce it.
echo "== pilot21_loss_phase_and_scheduler (stages b1, a x3, b2, c, d, b3)"
p21=pilot21_loss_phase_and_scheduler.py
rm -f pilot21_results.json pilot21_momenta.npz pilot21_kmin.json
python3 $p21 --stage b1 >> pilot21_run.log 2>&1
for task in diabetes resmlp digits; do
    case $task in
        diabetes) steps=1200; seeds=0,1,2 ;;
        resmlp)   steps=2000; seeds=0,1,2 ;;
        digits)   steps=1600; seeds=0,1,2,3,4 ;;
    esac
    python3 $p21 --stage a --task $task --steps $steps --seeds $seeds \
        --cond fp32,bf16,e4m3,e5m2,e5m2_s1035,ours,ours4 >> pilot21_run.log 2>&1
done
python3 $p21 --stage b2 > pilot21_run_b2.log 2>&1
python3 $p21 --stage c --task diabetes --steps 1200 --seeds 0,1,2 \
    --cond bf16,e4m3,e5m2_s1035 >> pilot21_run.log 2>&1
python3 $p21 --stage d > pilot21_run_d.log 2>&1
python3 $p21 --stage b3 --task diabetes --steps 1200 --seeds 0,1,2 \
    --cond psapA,psapB,psapC >> pilot21_run.log 2>&1
echo "== pilot21b_kmin (per-matrix step counts)"
python3 pilot21b_kmin.py

# pilot22 and pilot23 are also stage-driven; these are the invocations that
# produced results/pilot22_results.json and results/pilot23_results.json.
echo "== pilot22_participation_factor (the profile factor and its closure)"
python3 pilot22_participation_factor.py > pilot22_run.log 2>&1
echo "== pilot23_divergence_mechanism (window, census, counterfactuals, loop gain)"
p23=pilot23_divergence_mechanism.py
rm -f pilot23_results.json
python3 $p23 W > pilot23_run.log 2>&1
python3 $p23 E >> pilot23_run.log 2>&1
for c in e5m2 e5m2_s1035 e5m2_clamp e5m2_lr09; do
    python3 $p23 $c >> pilot23_run.log 2>&1
done
python3 $p23 B >> pilot23_run.log 2>&1
python3 $p23 D >> pilot23_run.log 2>&1

# pilot24 reads pilot22_results.json (../results/ in this layout) for the quartiles of
# Appendix E and writes the figure coordinates together with make_figure_data.py, which
# reads only JSON files; the figures in paper/figures/*.tex read nothing else.
echo "== pilot24_large_d_multiplicity (multiplicity to d = 512, quartiles, the ratchet bound)"
python3 pilot24_large_d_multiplicity.py > pilot24_run.log 2>&1
# pilot25 is the long census of §6.6: 1,200 steps x 25 seeds x 3 conditions x 2 tasks.
# It is resumable and parallel; the packaged log is the report of the second call.
echo "== pilot25_long_census (the hazard of a Muon call over 1,200 steps)"
rm -f pilot25_results.json
python3 pilot25_long_census.py --workers 2 > pilot25_run.log 2>&1
python3 pilot25_long_census.py --analyse >> pilot25_run.log 2>&1
echo "== make_figure_data (the coordinates of Figs. 1-3)"
python3 make_figure_data.py > make_figure_data_run.log 2>&1

echo "== static check of the paper sources"
cd ../paper && python3 validate.py && python3 validate_figures.py && python3 pagebudget.py
