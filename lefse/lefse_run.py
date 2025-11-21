#!/usr/bin/env python3

import os, sys, math, pickle, argparse
import logging
from pathlib import Path
from lefse import (init, load_data, get_class_means, test_kw_r, 
                   test_rep_wilcoxon_r, test_lda_r, test_svm, save_res)

# Configure logging
def setup_logging(verbose=False):
    """Setup logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('lefse_debug.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)

def inspect_input_file(input_file, max_lines=10):
    """Inspect and display first few lines of input file for debugging"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Inspecting input file structure: {input_file}")
    
    try:
        with open(input_file, 'r') as f:
            lines = [f.readline().strip() for _ in range(max_lines)]
        
        logger.info("First few rows of input file:")
        for i, line in enumerate(lines, 1):
            if line:
                # Show first 200 chars of each line
                preview = line[:200] + "..." if len(line) > 200 else line
                tabs = line.count('\t')
                logger.info(f"  Row {i} ({tabs} tabs): {preview}")
        
        # Analyze structure
        if len(lines) > 0:
            first_line_parts = lines[0].split('\t')
            logger.info(f"Number of columns: {len(first_line_parts)}")
            logger.info(f"First column value: '{first_line_parts[0]}'")
            
            if len(lines) > 1:
                second_line_parts = lines[1].split('\t')
                logger.info(f"Second row first column: '{second_line_parts[0]}'")
        
    except Exception as e:
        logger.warning(f"Could not inspect input file: {e}")

def validate_input_file(input_file):
    """Validate input file exists and is readable"""
    logger = logging.getLogger(__name__)
    
    if not input_file:
        logger.error("Input file path is empty")
        raise ValueError("Input file path cannot be empty")
    
    input_path = Path(input_file)
    
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if not input_path.is_file():
        logger.error(f"Input path is not a file: {input_file}")
        raise ValueError(f"Input path is not a file: {input_file}")
    
    if not os.access(input_file, os.R_OK):
        logger.error(f"Input file is not readable: {input_file}")
        raise PermissionError(f"Cannot read input file: {input_file}")
    
    file_size = input_path.stat().st_size
    logger.info(f"Input file validated: {input_file} ({file_size} bytes)")
    
    # Inspect file structure
    inspect_input_file(input_file)
    
    return True

def validate_output_path(output_file):
    """Validate output file path is writable"""
    logger = logging.getLogger(__name__)
    
    if not output_file:
        logger.error("Output file path is empty")
        raise ValueError("Output file path cannot be empty")
    
    output_path = Path(output_file)
    output_dir = output_path.parent
    
    if not output_dir.exists():
        logger.warning(f"Output directory does not exist, attempting to create: {output_dir}")
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            raise
    
    if output_path.exists() and not os.access(output_file, os.W_OK):
        logger.error(f"Output file exists but is not writable: {output_file}")
        raise PermissionError(f"Cannot write to output file: {output_file}")
    
    logger.info(f"Output path validated: {output_file}")
    return True

def validate_params(params):
    """Validate parameter values"""
    logger = logging.getLogger(__name__)
    logger.info("Validating parameters...")
    
    # Validate alpha values
    if not 0 < params['anova_alpha'] <= 1:
        raise ValueError(f"anova_alpha must be between 0 and 1, got: {params['anova_alpha']}")
    
    if not 0 < params['wilcoxon_alpha'] <= 1:
        raise ValueError(f"wilcoxon_alpha must be between 0 and 1, got: {params['wilcoxon_alpha']}")
    
    # Validate LDA threshold
    if params['lda_abs_th'] < 0 and params['lda_abs_th'] != -1:
        logger.warning(f"Unusual lda_abs_th value: {params['lda_abs_th']}")
    
    # Validate bootstrap parameters
    if params['n_boots'] < 1:
        raise ValueError(f"n_boots must be positive, got: {params['n_boots']}")
    
    if not 0 < params['f_boots'] <= 1:
        raise ValueError(f"f_boots must be between 0 and 1, got: {params['f_boots']}")
    
    # Validate min_c
    if params['min_c'] < 1:
        raise ValueError(f"min_c must be positive, got: {params['min_c']}")
    
    logger.info("Parameter validation passed")
    logger.debug(f"Parameters: {params}")
    return True

def diagnose_class_structure(cls, class_sl, feats):
    """Diagnose class structure and data quality for LDA"""
    logger = logging.getLogger(__name__)
    
    logger.info("Diagnosing class structure for LDA:")
    
    # class_sl is a dict mapping class names to (start_idx, end_idx) tuples
    logger.info(f"  - Class labels found: {list(class_sl.keys())}")
    logger.info(f"  - Number of unique classes: {len(class_sl)}")
    
    # Check sample distribution per class
    class_samples = {}
    for class_label, (start_idx, end_idx) in class_sl.items():
        num_samples = end_idx - start_idx
        class_samples[class_label] = num_samples
        logger.info(f"  - Class '{class_label}': {num_samples} samples (indices {start_idx} to {end_idx})")
    
    # Check if we have enough classes for LDA
    if len(class_sl) < 2:
        logger.error("LDA requires at least 2 classes")
        return False
    
    # Check if we have enough samples per class
    min_samples = min(class_samples.values())
    max_samples = max(class_samples.values())
    
    if min_samples < 2:
        logger.error(f"At least one class has fewer than 2 samples (min: {min_samples})")
        logger.error("")
        logger.error("=" * 70)
        logger.error("DIAGNOSIS: INSUFFICIENT SAMPLES PER CLASS")
        logger.error("=" * 70)
        logger.error("")
        logger.error("Your classes have very few samples:")
        for class_label, count in sorted(class_samples.items(), key=lambda x: x[1]):
            logger.error(f"  - {class_label}: {count} samples")
        logger.error("")
        logger.error("LDA requires at least 2 samples per class for statistical analysis.")
        logger.error("")
        logger.error("POSSIBLE CAUSES:")
        logger.error("1. The input data doesn't have enough biological replicates")
        logger.error("2. The wrong metadata row was specified as the class")
        logger.error("3. Samples were filtered out during preprocessing")
        logger.error("")
        logger.error("SOLUTIONS:")
        logger.error("- Verify you used the correct -c, -s, -u parameters in format_input.py")
        logger.error("- Check your original data file has enough samples per condition")
        logger.error("- Consider using a simpler comparison (e.g., just 2 groups instead of 3)")
        logger.error("=" * 70)
        return False
    
    # Warn about imbalanced classes
    if max_samples > min_samples * 5:
        logger.warning(f"Highly imbalanced classes detected:")
        logger.warning(f"  - Minimum: {min_samples} samples")
        logger.warning(f"  - Maximum: {max_samples} samples")
        logger.warning(f"  - Ratio: {max_samples/min_samples:.1f}x")
        logger.warning("This may affect LDA performance")
    
    # Check feature statistics
    logger.info(f"  - Number of features: {len(feats)}")
    
    # Sample a few features to check for variance
    sample_size = min(5, len(feats))
    sample_feats = list(feats.items())[:sample_size]
    
    for feat_name, feat_values in sample_feats:
        # feat_values can be either a dict or a list depending on the data structure
        if isinstance(feat_values, dict):
            unique_vals = len(set(feat_values.values()))
            non_zero = sum(1 for v in feat_values.values() if v != 0)
            total = len(feat_values)
        else:  # list or other iterable
            unique_vals = len(set(feat_values))
            non_zero = sum(1 for v in feat_values if v != 0)
            total = len(feat_values)
        logger.debug(f"  - Feature '{feat_name}':")
        logger.debug(f"      {unique_vals} unique values, {non_zero}/{total} non-zero")
    
    logger.info(f"✓ Class structure is valid for LDA")
    return True

def read_params(args):
    parser = argparse.ArgumentParser(description='LEfSe 1.1.01 with Enhanced Debugging')
    parser.add_argument('input_file', metavar='INPUT_FILE', type=str, help="the input file")
    parser.add_argument('output_file', metavar='OUTPUT_FILE', type=str,
                help="the output file containing the data for the visualization module")
    parser.add_argument('-o',dest="out_text_file", metavar='str', type=str, default="",
                help="set the file for exporting the result (only concise textual form)")
    parser.add_argument('-a',dest="anova_alpha", metavar='float', type=float, default=0.05,
                help="set the alpha value for the Anova test (default 0.05)")
    parser.add_argument('-w',dest="wilcoxon_alpha", metavar='float', type=float, default=0.05,
                help="set the alpha value for the Wilcoxon test (default 0.05)")
    parser.add_argument('-l',dest="lda_abs_th", metavar='float', type=float, default=2.0,
                help="set the threshold on the absolute value of the logarithmic LDA score (default 2.0)")
    parser.add_argument('--nlogs',dest="nlogs", metavar='int', type=int, default=3,
        help="max log ingluence of LDA coeff")
    parser.add_argument('--verbose',dest="verbose", metavar='int', choices=[0,1], type=int, default=0,
        help="verbose execution (default 0)")
    parser.add_argument('--wilc',dest="wilc", metavar='int', choices=[0,1], type=int, default=1,
        help="wheter to perform the Wicoxon step (default 1)")
    parser.add_argument('-r',dest="rank_tec", metavar='str', choices=['lda','svm'], type=str, default='lda',
        help="select LDA or SVM for effect size (default LDA)")
    parser.add_argument('--svm_norm',dest="svm_norm", metavar='int', choices=[0,1], type=int, default=1,
        help="whether to normalize the data in [0,1] for SVM feature waiting (default 1 strongly suggested)")
    parser.add_argument('-b',dest="n_boots", metavar='int', type=int, default=30,
                help="set the number of bootstrap iteration for LDA (default 30)")
    parser.add_argument('-e',dest="only_same_subcl", metavar='int', type=int, default=0,
                help="set whether perform the wilcoxon test only among the subclasses with the same name (default 0)")
    parser.add_argument('-c',dest="curv", metavar='int', type=int, default=0,
                help="set whether perform the wilcoxon test ing the Curtis's approach [BETA VERSION] (default 0)")
    parser.add_argument('-f',dest="f_boots", metavar='float', type=float, default=0.67,
                help="set the subsampling fraction value for each bootstrap iteration (default 0.66666)")
    parser.add_argument('-s',dest="strict", choices=[0,1,2], type=int, default=0,
                help="set the multiple testing correction options. 0 no correction (more strict, default), 1 correction for independent comparisons, 2 correction for dependent comparison")
    parser.add_argument('--min_c',dest="min_c", metavar='int', type=int, default=10,
                help="minimum number of samples per subclass for performing wilcoxon test (default 10)")
    parser.add_argument('-t',dest="title", metavar='str', type=str, default="",
                help="set the title of the analysis (default input file without extension)")
    parser.add_argument('-y',dest="multiclass_strat", choices=[0,1], type=int, default=0,
                help="(for multiclass tasks) set whether the test is performed in a one-against-one ( 1 - more strict!) or in a one-against-all setting ( 0 - less strict) (default 0)")
    args = parser.parse_args()

    params = vars(args)
    if params['title'] == "":
        params['title'] = params['input_file'].split("/")[-1].split('.')[0]

    return params


def lefse_run():
    """Main LEfSe execution with debugging and validation"""
    params = None
    logger = None
    
    try:
        # Initialize
        init()
        params = read_params(sys.argv)
        
        # Setup logging
        logger = setup_logging(params['verbose'])
        logger.info("="*60)
        logger.info("LEfSe Analysis Started")
        logger.info("="*60)
        
        # Validate inputs
        logger.info("Step 1: Validating input files and parameters...")
        validate_input_file(params['input_file'])
        validate_output_path(params['output_file'])
        validate_params(params)
        
        # Load data
        logger.info("Step 2: Loading data...")
        feats, cls, class_sl, subclass_sl, class_hierarchy = load_data(params['input_file'])
        
        logger.info(f"Data loaded successfully:")
        logger.info(f"  - Number of features: {len(feats)}")
        logger.info(f"  - Number of classes: {len(cls)}")
        logger.info(f"  - Classes: {list(cls.keys())}")
        logger.debug(f"  - Class hierarchy: {class_hierarchy}")
        
        # Detailed inspection of cls structure
        logger.debug("Detailed cls structure:")
        for key, value in cls.items():
            if isinstance(value, (list, tuple)):
                logger.debug(f"  - cls['{key}']: {len(value)} items, first 5: {value[:5]}")
            else:
                logger.debug(f"  - cls['{key}']: {value}")
        
        # Detailed inspection of class_sl
        logger.debug("Detailed class_sl structure:")
        logger.debug(f"  - class_sl keys (first 10): {list(class_sl.keys())[:10]}")
        logger.debug(f"  - class_sl values (first 5): {list(class_sl.values())[:5]}")
        
        # Detailed inspection of subclass_sl
        logger.debug("Detailed subclass_sl structure:")
        logger.debug(f"  - subclass_sl keys (first 10): {list(subclass_sl.keys())[:10]}")
        logger.debug(f"  - subclass_sl values (first 5): {list(subclass_sl.values())[:5]}")
        
        if len(feats) == 0:
            logger.error("No features found in input file")
            raise ValueError("Input file contains no features")
        
        # Get class means
        logger.info("Step 3: Computing class means...")
        kord, cls_means = get_class_means(class_sl, feats)
        logger.info(f"Class means computed for {len(cls_means)} features")
        
        # Kruskal-Wallis and Wilcoxon tests
        logger.info("Step 4: Running Kruskal-Wallis tests...")
        wilcoxon_res = {}
        kw_n_ok = 0
        nf = 0
        kw_rejected = 0
        wilc_rejected = 0
        
        total_features = len(feats)
        feat_items = list(feats.items())
        
        for feat_name, feat_values in feat_items:
            nf += 1
            if params['verbose']:
                logger.debug(f"Testing feature {nf}/{total_features}: {feat_name}")
            
            # Kruskal-Wallis test
            try:
                kw_ok, pv = test_kw_r(cls, feat_values, params['anova_alpha'], 
                                      sorted(cls.keys()))
                
                if not kw_ok:
                    if params['verbose']: 
                        logger.debug(f"  KW rejected (p-value: {pv:.6f})")
                    del feats[feat_name]
                    wilcoxon_res[feat_name] = "-"
                    kw_rejected += 1
                    continue
                    
                if params['verbose']: 
                    logger.debug(f"  KW passed (p-value: {pv:.6f})")
                
                kw_n_ok += 1
                
                # Wilcoxon test
                if not params['wilc']: 
                    continue
                
                res_wilcoxon_rep = test_rep_wilcoxon_r(
                    subclass_sl, class_hierarchy, feat_values,
                    params['wilcoxon_alpha'], params['multiclass_strat'],
                    params['strict'], feat_name, params['min_c'],
                    params['only_same_subcl'], params['curv']
                )
                
                wilcoxon_res[feat_name] = str(pv) if res_wilcoxon_rep else "-"
                
                if not res_wilcoxon_rep:
                    if params['verbose']: 
                        logger.debug("  Wilcoxon rejected")
                    del feats[feat_name]
                    wilc_rejected += 1
                elif params['verbose']: 
                    logger.debug("  Wilcoxon passed")
                    
            except Exception as e:
                logger.error(f"Error testing feature {feat_name}: {e}")
                if feat_name in feats:
                    del feats[feat_name]
                wilcoxon_res[feat_name] = "-"
                continue
        
        logger.info(f"Statistical testing complete:")
        logger.info(f"  - KW passed: {kw_n_ok}/{total_features}")
        logger.info(f"  - KW rejected: {kw_rejected}/{total_features}")
        logger.info(f"  - Wilcoxon rejected: {wilc_rejected}/{kw_n_ok}")
        logger.info(f"  - Features remaining: {len(feats)}")
        
        # LDA/SVM ranking
        if len(feats) > 0:
            logger.info("Step 5: Computing effect sizes...")
            print(f"Number of significantly discriminative features: {len(feats)} "
                  f"({kw_n_ok}) before internal wilcoxon")
            
            # Diagnose class structure before attempting LDA
            if not diagnose_class_structure(cls, class_sl, feats):
                logger.error("Class structure validation failed - cannot proceed with LDA")
                raise ValueError("Invalid class structure for LDA computation")
            
            # Validate data before LDA/SVM
            logger.debug(f"Validating data for effect size computation:")
            logger.debug(f"  - Features: {len(feats)}")
            logger.debug(f"  - Classes in cls: {list(cls.keys())}")
            logger.debug(f"  - Classes in class_sl: {set([v for vals in class_sl.values() for v in vals])}")
            
            # Check class distribution
            class_counts = {}
            for class_name, samples in cls.items():
                class_counts[class_name] = len(samples)
                logger.debug(f"  - Class '{class_name}': {len(samples)} samples")
            
            # Check for sufficient samples per class
            min_samples = min(class_counts.values())
            logger.debug(f"  - Minimum samples per class: {min_samples}")
            
            if min_samples < 2:
                logger.error(f"Insufficient samples for LDA: at least one class has < 2 samples")
                logger.error(f"Class distribution: {class_counts}")
                raise ValueError("LDA requires at least 2 samples per class")
            
            # Check bootstrap fraction feasibility
            min_bootstrap_size = int(min_samples * params['f_boots'])
            logger.debug(f"  - Min bootstrap size (f_boots={params['f_boots']}): {min_bootstrap_size}")
            
            if min_bootstrap_size < 1:
                logger.warning(f"Bootstrap fraction too small, may cause issues")
            
            if params['lda_abs_th'] < 0.0:
                logger.info("Skipping LDA (threshold < 0)")
                lda_res = dict([(k, 0.0) for k, v in feats.items()])
                lda_res_th = dict([(k, v) for k, v in feats.items()])
            else:
                try:
                    if params['rank_tec'] == 'lda':
                        logger.info(f"Running LDA with {params['n_boots']} bootstraps...")
                        logger.debug(f"LDA parameters:")
                        logger.debug(f"  - n_boots: {params['n_boots']}")
                        logger.debug(f"  - f_boots: {params['f_boots']}")
                        logger.debug(f"  - lda_abs_th: {params['lda_abs_th']}")
                        logger.debug(f"  - nlogs: {params['nlogs']}")
                        
                        # Detailed data inspection before LDA
                        logger.debug(f"Data being passed to LDA:")
                        logger.debug(f"  - cls type: {type(cls)}, keys: {list(cls.keys())}")
                        logger.debug(f"  - feats type: {type(feats)}, num features: {len(feats)}")
                        logger.debug(f"  - class_sl type: {type(class_sl)}, classes: {list(class_sl.keys())}")
                        
                        # Check class sizes
                        for class_name, (start, end) in class_sl.items():
                            logger.debug(f"  - Class {class_name}: {end - start} samples")
                        
                        # Check feature dimensions
                        first_feat = list(feats.values())[0]
                        logger.debug(f"  - Feature data type: {type(first_feat)}")
                        logger.debug(f"  - Feature length: {len(first_feat)}")
                        
                        # Calculate expected bootstrap size
                        min_class_size = min(end - start for start, end in class_sl.values())
                        bootstrap_size = int(min_class_size * params['f_boots'])
                        logger.debug(f"  - Min class size: {min_class_size}")
                        logger.debug(f"  - Expected bootstrap samples per class: {bootstrap_size}")
                        
                        if bootstrap_size < 2:
                            logger.error(f"Bootstrap size ({bootstrap_size}) too small!")
                            logger.error(f"With f_boots={params['f_boots']} and min class size {min_class_size},")
                            logger.error(f"bootstrap will have < 2 samples per class")
                            logger.error(f"Try increasing -f parameter (e.g., -f 0.9) or reducing features")
                            raise ValueError(f"Bootstrap size too small: {bootstrap_size}")
                        
                        logger.info(f"Calling test_lda_r...")
                        lda_res, lda_res_th = test_lda_r(
                            cls, feats, class_sl, params['n_boots'],
                            params['f_boots'], params['lda_abs_th'],
                            1e-7, params['nlogs']
                        )
                        logger.info(f"test_lda_r returned successfully")
                        
                        # Validate LDA results
                        if lda_res is None or lda_res_th is None:
                            logger.error("LDA returned None - computation failed")
                            logger.error("This usually indicates:")
                            logger.error("  1. Insufficient sample diversity within classes")
                            logger.error("  2. Perfect separation between classes")
                            logger.error("  3. Collinear features")
                            logger.error("  4. R computation error")
                            logger.error("")
                            logger.error("DIAGNOSTIC INFO:")
                            logger.error(f"  - Number of features: {len(feats)}")
                            logger.error(f"  - Number of samples: {len(first_feat)}")
                            logger.error(f"  - Number of classes: {len(class_sl)}")
                            logger.error(f"  - Bootstrap samples per class: {bootstrap_size}")
                            logger.error("")
                            logger.error("SUGGESTIONS:")
                            logger.error("  1. Try with fewer features (only top discriminative ones)")
                            logger.error("  2. Increase bootstrap fraction: -f 0.9")
                            logger.error("  3. Reduce bootstrap iterations: -b 10")
                            logger.error("  4. Compare only 2 groups instead of 3")
                            logger.error("  5. Skip LDA and use KW results only: -l -1")
                            raise ValueError("LDA computation failed - check data quality and class distribution")
                        
                    elif params['rank_tec'] == 'svm':
                        logger.info(f"Running SVM with {params['n_boots']} bootstraps...")
                        logger.debug(f"SVM parameters:")
                        logger.debug(f"  - n_boots: {params['n_boots']}")
                        logger.debug(f"  - f_boots: {params['f_boots']}")
                        logger.debug(f"  - lda_abs_th: {params['lda_abs_th']}")
                        logger.debug(f"  - svm_norm: {params['svm_norm']}")
                        
                        lda_res, lda_res_th = test_svm(
                            cls, feats, class_sl, params['n_boots'],
                            params['f_boots'], params['lda_abs_th'],
                            0.0, params['svm_norm']
                        )
                        
                        if lda_res is None or lda_res_th is None:
                            logger.error("SVM returned None - computation failed")
                            raise ValueError("SVM computation failed - check data quality")
                        
                    else:
                        lda_res = dict([(k, 0.0) for k, v in feats.items()])
                        lda_res_th = dict([(k, v) for k, v in feats.items()])
                    
                    logger.info(f"Effect size computation complete: {len(lda_res_th)} features above threshold")
                    logger.debug(f"LDA results summary:")
                    logger.debug(f"  - Total features scored: {len(lda_res)}")
                    logger.debug(f"  - Features above threshold: {len(lda_res_th)}")
                    
                except Exception as e:
                    logger.error(f"Error during effect size computation: {e}")
                    logger.error(f"This may be due to:")
                    logger.error(f"  - Insufficient variance in features")
                    logger.error(f"  - Too few samples in one or more classes")
                    logger.error(f"  - Data quality issues")
                    raise
        else:
            logger.warning("No features passed statistical tests")
            print(f"Number of significantly discriminative features: {len(feats)} "
                  f"({kw_n_ok}) before internal wilcoxon")
            print("No features with significant differences between the two classes")
            lda_res, lda_res_th = {}, {}
        
        # Prepare output
        logger.info("Step 6: Preparing output...")
        outres = {}
        outres['lda_res_th'] = lda_res_th
        outres['lda_res'] = lda_res
        outres['cls_means'] = cls_means
        outres['cls_means_kord'] = kord
        outres['wilcox_res'] = wilcoxon_res
        
        print(f"Number of discriminative features with abs LDA score > "
              f"{params['lda_abs_th']}: {len(lda_res_th)}")
        
        # Save results
        logger.info("Step 7: Saving results...")
        save_res(outres, params["output_file"])
        logger.info(f"Results saved to: {params['output_file']}")
        
        logger.info("="*60)
        logger.info("LEfSe Analysis Completed Successfully")
        logger.info("="*60)
        
    except FileNotFoundError as e:
        if logger:
            logger.error(f"File not found: {e}")
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
        
    except PermissionError as e:
        if logger:
            logger.error(f"Permission error: {e}")
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
        
    except ValueError as e:
        if logger:
            logger.error(f"Invalid value: {e}")
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        if logger:
            logger.exception(f"Unexpected error: {e}")
        print(f"ERROR: Unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    lefse_run()