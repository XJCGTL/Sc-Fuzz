# Copyright 2023 Flavien Solt, ETH Zurich.
# Licensed under the General Public License, Version 3.0, see LICENSE for details.
# SPDX-License-Identifier: GPL-3.0-only

from cascade.cfinstructionclasses import *
from cascade.toleratebugs import is_tolerate_cva6_fdivs_flags, is_tolerate_vexriscv_imprecise_fcvt, is_tolerate_vexriscv_fmin, is_tolerate_vexriscv_double_to_float, is_tolerate_vexriscv_dependent_single_precision, is_tolerate_vexriscv_dependent_fle_feq_ret1, is_tolerate_vexriscv_dependent_flt_ret0, is_tolerate_vexriscv_sqrt, is_tolerate_vexriscv_muldiv_conversion, is_tolerate_cva6_single_precision, is_tolerate_cva6_division
from cascade.util import ISAInstrClass, IntRegIndivState, INSTRUCTIONS_BY_ISA_CLASS
from params.fuzzparams import NUM_MIN_FREE_INTREGS

from copy import copy
from collections import defaultdict
import random

# For a given ISAInstrClass, this module helps picking an instruction type.

###
# Helper functions
###

# INSTRTYPE_INITIAL_RELATIVE_WEIGHTS[ISAInstrClass]["add"] = float. Weights will be normalized by instr class.
INSTRTYPE_INITIAL_RELATIVE_WEIGHTS = {
    # For the moment, give all instructions inside the same ISA class the same appearance chance.
    curr_key: dict.fromkeys(curr_instrs, 1) for curr_key, curr_instrs in INSTRUCTIONS_BY_ISA_CLASS.items()
}

INSTR_EPSILON_EXPLORATION = 0.10
INSTR_MIN_POSITIVE_WEIGHT = 1e-9
INSTR_BUG_WEIGHT = 0.55
INSTR_COVERAGE_WEIGHT = 0.30
INSTR_NOVELTY_WEIGHT = 0.15
INSTR_SELECTION_DECAY = 0.03

# Useful for time to bug evaluation
def forbid_vexriscv_ops(keys_and_weights_dict):
    keys_and_weights_dict_ret = copy(keys_and_weights_dict)
    # Double precision
    keys_and_weights_dict_ret["fsgnj.d"] = 0
    keys_and_weights_dict_ret["fsgnjn.d"] = 0
    keys_and_weights_dict_ret["fsgnjx.d"] = 0
    keys_and_weights_dict_ret["fnmadd.s"] = 0
    keys_and_weights_dict_ret["fmadd.s"] = 0
    keys_and_weights_dict_ret["fnmsub.s"] = 0
    keys_and_weights_dict_ret["fmsub.s"] = 0

    # Single precision
    keys_and_weights_dict_ret["fsgnj.s"] = 0
    keys_and_weights_dict_ret["fsgnjn.s"] = 0
    keys_and_weights_dict_ret["fsgnjx.s"] = 0
    keys_and_weights_dict_ret["fnmadd.d"] = 0
    keys_and_weights_dict_ret["fmadd.d"] = 0
    keys_and_weights_dict_ret["fnmsub.d"] = 0
    keys_and_weights_dict_ret["fmsub.d"] = 0

    if is_tolerate_vexriscv_imprecise_fcvt() or is_tolerate_vexriscv_fmin() or is_tolerate_vexriscv_double_to_float() or is_tolerate_vexriscv_dependent_single_precision() or is_tolerate_vexriscv_dependent_fle_feq_ret1() or is_tolerate_vexriscv_dependent_flt_ret0() or is_tolerate_vexriscv_sqrt() or is_tolerate_vexriscv_muldiv_conversion():
        keys_and_weights_dict_ret["fcvt.w.s"] = 0
        keys_and_weights_dict_ret["fcvt.wu.s"] = 0
        keys_and_weights_dict_ret["fmin.s"] = 0
        keys_and_weights_dict_ret["fmax.s"] = 0
        keys_and_weights_dict_ret["fsqrt.s"] = 0
        keys_and_weights_dict_ret["fmul.s"] = 0
        keys_and_weights_dict_ret["fmul.d"] = 0
        keys_and_weights_dict_ret["fadd.s"] = 0
        keys_and_weights_dict_ret["fsub.s"] = 0
        keys_and_weights_dict_ret["fdiv.s"] = 0
        keys_and_weights_dict_ret["fdiv.d"] = 0
        keys_and_weights_dict_ret["flt.s"] = 0
        keys_and_weights_dict_ret["fle.s"] = 0
        keys_and_weights_dict_ret["feq.s"] = 0
        keys_and_weights_dict_ret["fcvt.l.s"] = 0
        keys_and_weights_dict_ret["fcvt.lu.s"] = 0
        keys_and_weights_dict_ret["fcvt.s.w"] = 0
        keys_and_weights_dict_ret["fcvt.s.wu"] = 0
        keys_and_weights_dict_ret["fcvt.s.l"] = 0
        keys_and_weights_dict_ret["fcvt.s.lu"] = 0
        keys_and_weights_dict_ret["fcvt.d.s"] = 0
        keys_and_weights_dict_ret["fmin.d"] = 0
        keys_and_weights_dict_ret["fmax.d"] = 0

        if is_tolerate_vexriscv_imprecise_fcvt() or is_tolerate_vexriscv_fmin() or is_tolerate_vexriscv_double_to_float() or is_tolerate_vexriscv_dependent_single_precision() or is_tolerate_vexriscv_dependent_fle_feq_ret1() or is_tolerate_vexriscv_dependent_flt_ret0() or is_tolerate_vexriscv_sqrt():
            keys_and_weights_dict_ret["fcvt.w.s"] = keys_and_weights_dict["fcvt.w.s"]
            keys_and_weights_dict_ret["fcvt.wu.s"] = keys_and_weights_dict["fcvt.wu.s"]
            keys_and_weights_dict_ret["fcvt.l.s"] = keys_and_weights_dict["fcvt.l.s"]
            keys_and_weights_dict_ret["fcvt.lu.s"] = keys_and_weights_dict["fcvt.lu.s"]
            keys_and_weights_dict_ret["fcvt.d.s"] = keys_and_weights_dict["fcvt.d.s"]

        if is_tolerate_vexriscv_imprecise_fcvt():
            keys_and_weights_dict_ret["fcvt.s.w"] = keys_and_weights_dict["fcvt.s.w"]
            keys_and_weights_dict_ret["fcvt.s.wu"] = keys_and_weights_dict["fcvt.s.wu"]
            keys_and_weights_dict_ret["fcvt.s.l"] = keys_and_weights_dict["fcvt.s.l"]
            keys_and_weights_dict_ret["fcvt.s.lu"] = keys_and_weights_dict["fcvt.s.lu"]
        if is_tolerate_vexriscv_fmin():
            keys_and_weights_dict_ret["fmin.s"] = keys_and_weights_dict["fmin.s"]
            keys_and_weights_dict_ret["fmin.d"] = keys_and_weights_dict["fmin.d"]
        if is_tolerate_vexriscv_double_to_float():
            keys_and_weights_dict_ret["fcvt.d.s"] = keys_and_weights_dict["fcvt.d.s"]
        if is_tolerate_vexriscv_dependent_single_precision():
            keys_and_weights_dict_ret["fmul.s"] = keys_and_weights_dict["fmul.s"]
            keys_and_weights_dict_ret["fadd.s"] = keys_and_weights_dict["fadd.s"]
            keys_and_weights_dict_ret["fsub.s"] = keys_and_weights_dict["fsub.s"]
            keys_and_weights_dict_ret["fdiv.s"] = keys_and_weights_dict["fdiv.s"]
        if is_tolerate_vexriscv_dependent_fle_feq_ret1():
            keys_and_weights_dict_ret["fle.s"] = keys_and_weights_dict["fle.s"]
            keys_and_weights_dict_ret["fle.d"] = keys_and_weights_dict["fle.d"]
            keys_and_weights_dict_ret["feq.s"] = keys_and_weights_dict["feq.s"]
            keys_and_weights_dict_ret["feq.d"] = keys_and_weights_dict["feq.d"]
        if is_tolerate_vexriscv_dependent_flt_ret0():
            keys_and_weights_dict_ret["flt.s"] = keys_and_weights_dict["flt.s"]
            keys_and_weights_dict_ret["flt.d"] = keys_and_weights_dict["flt.d"]
        if is_tolerate_vexriscv_sqrt():
            keys_and_weights_dict_ret["fsqrt.s"] = keys_and_weights_dict["fsqrt.s"]
            keys_and_weights_dict_ret["fsqrt.d"] = keys_and_weights_dict["fsqrt.d"]
    
        if is_tolerate_vexriscv_muldiv_conversion():
            keys_and_weights_dict_ret["fmul.s"] = keys_and_weights_dict["fmul.s"]
            keys_and_weights_dict_ret["fmul.d"] = keys_and_weights_dict["fmul.d"]
            keys_and_weights_dict_ret["fdiv.s"] = keys_and_weights_dict["fdiv.s"]
            keys_and_weights_dict_ret["fdiv.d"] = keys_and_weights_dict["fdiv.d"]

    return keys_and_weights_dict_ret

def _feedback_value_for_instr(feedback_map, isaclass, instrstr):
    if not isinstance(feedback_map, dict):
        return 0.0

    # 1) Flat map: {"add": ...}
    if instrstr in feedback_map:
        try:
            return float(feedback_map[instrstr])
        except (TypeError, ValueError):
            return 0.0

    # 2) Nested map: {ISAInstrClass.ALU: {"add": ...}} or {"ALU": {"add": ...}}
    candidates = [isaclass]
    if hasattr(isaclass, "name"):
        candidates.append(isaclass.name)
    candidates.append(str(isaclass))
    for curr_key in candidates:
        if curr_key in feedback_map and isinstance(feedback_map[curr_key], dict) and instrstr in feedback_map[curr_key]:
            try:
                return float(feedback_map[curr_key][instrstr])
            except (TypeError, ValueError):
                return 0.0

    return 0.0

def _apply_dynamic_instr_scores(keys_and_weights_dict, isaclass, fuzzerstate):
    bug_scores = getattr(fuzzerstate, "instr_bug_scores", {})
    coverage_scores = getattr(fuzzerstate, "instr_coverage_scores", {})
    novelty_scores = getattr(fuzzerstate, "instr_novelty_scores", {})
    selection_counts = getattr(fuzzerstate, "instr_selection_counts", {})

    for instrstr in keys_and_weights_dict:
        base_weight = keys_and_weights_dict[instrstr]
        if base_weight <= 0:
            continue

        bug_score = max(0.0, _feedback_value_for_instr(bug_scores, isaclass, instrstr))
        coverage_score = max(0.0, _feedback_value_for_instr(coverage_scores, isaclass, instrstr))
        novelty_score = max(0.0, _feedback_value_for_instr(novelty_scores, isaclass, instrstr))
        pick_count = max(0.0, _feedback_value_for_instr(selection_counts, isaclass, instrstr))

        reward_multiplier = 1.0 + \
            INSTR_BUG_WEIGHT * bug_score + \
            INSTR_COVERAGE_WEIGHT * coverage_score + \
            INSTR_NOVELTY_WEIGHT * novelty_score
        decay = 1.0 / (1.0 + INSTR_SELECTION_DECAY * pick_count)
        keys_and_weights_dict[instrstr] = max(base_weight * reward_multiplier * decay, INSTR_MIN_POSITIVE_WEIGHT)

def _pick_instr_with_exploration(keys_and_weights_dict, isaclass):
    positive_instrs = [instrstr for instrstr, curr_weight in keys_and_weights_dict.items() if curr_weight > 0]
    if not positive_instrs:
        # Keep the fuzzer alive even under overly restrictive dynamic/bug filters.
        fallback_instrs = [instrstr for instrstr, curr_weight in INSTRTYPE_INITIAL_RELATIVE_WEIGHTS[isaclass].items() if curr_weight > 0]
        if DO_ASSERT:
            assert len(fallback_instrs), f"No instruction candidate available for ISA class {isaclass}."
        return random.choice(fallback_instrs)

    if random.random() < INSTR_EPSILON_EXPLORATION:
        return random.choice(positive_instrs)

    return random.choices(positive_instrs, weights=[keys_and_weights_dict[curr_key] for curr_key in positive_instrs])[0]

###
# Exposed functions
###

# @param isaclass
# @return a CFInstruction object or a placeholder object
def gen_next_instrstr_from_isaclass(isaclass: ISAInstrClass, fuzzerstate) -> str:
    # This global may prevent from copying the dict
    global INSTRTYPE_INITIAL_RELATIVE_WEIGHTS

    keys_and_weights_dict = defaultdict(int, INSTRTYPE_INITIAL_RELATIVE_WEIGHTS[isaclass])

    # No floating point sign injection
    if "vexriscv" in fuzzerstate.design_name:
        keys_and_weights_dict = forbid_vexriscv_ops(keys_and_weights_dict)

    if isaclass == ISAInstrClass.SPECIAL:
        fuzzerstate.special_instrs_count += 1

    if fuzzerstate.design_name == "cva6":
        # Double precision
        keys_and_weights_dict["fsqrt.d"] = 0
        if not is_tolerate_cva6_division():
            keys_and_weights_dict["fdiv.d"] = 0

        # Single precision
        if not is_tolerate_cva6_single_precision():
            keys_and_weights_dict["fsqrt.s"] = 0
            keys_and_weights_dict["fdiv.s"] = 0
            keys_and_weights_dict["fcvt.d.s"] = 0
            keys_and_weights_dict["fcvt.s.d"] = 0
            keys_and_weights_dict["fcvt.wu.d"] = 0
            
        if not is_tolerate_cva6_fdivs_flags():
            keys_and_weights_dict["fdiv.s"] = 0


    if DO_ASSERT:
        assert isaclass != ISAInstrClass.REGFSM   , "ISAInstrClass.REGFSM must be treated separately"
        assert isaclass != ISAInstrClass.FPUFSM   , "ISAInstrClass.FPUFSM must be treated separately"
        assert isaclass != ISAInstrClass.EXCEPTION, "ISAInstrClass.EXCEPTION must be treated separately"
        assert isaclass != ISAInstrClass.TVECFSM  , "ISAInstrClass.TVECFSM must be treated separately"
        assert isaclass != ISAInstrClass.PPFSM    , "ISAInstrClass.PPFSM must be treated separately"
        assert isaclass != ISAInstrClass.EPCFSM   , "ISAInstrClass.EPCFSM must be treated separately"

    _apply_dynamic_instr_scores(keys_and_weights_dict, isaclass, fuzzerstate)
    return _pick_instr_with_exploration(keys_and_weights_dict, isaclass)
