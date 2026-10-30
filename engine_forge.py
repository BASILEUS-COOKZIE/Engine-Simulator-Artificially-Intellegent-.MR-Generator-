#!/usr/bin/env python3
"""engine_forge: generate engine-sim (.mr) engine files from a prompt or a JSON spec.

Usage
  python engine_forge.py --prompt "v12 6.0L 8000rpm fire in pairs" -o v12.mr
  python engine_forge.py --prompt "twin row radial 14 tapering cylinders 12000rpm" -o r14.mr
  python engine_forge.py --spec my_engine.json -o engine.mr
  python engine_forge.py --example            # print a full example spec
  python engine_forge.py --verify engine.mr   # check firing/journal/cam/wire consistency
  python engine_forge.py --llm --prompt "..." -o out.mr   # let Claude turn free text into a spec (needs ANTHROPIC_API_KEY)

Every generated engine passes the built-in consistency check (every cylinder fires at its TDC by the rule
journal angle = firing angle + bank angle mod 360, cam lobe offset = firing angle, every wire connected once).
Tuning numbers are estimates scaled from engines tuned in engine-sim by hand. They are starting points, not
measured data, and the generator cannot run engine-sim to score the result.
"""
import argparse
import json
import math
import os
import re
import sys

HEAD_AND_CAM = r'''private node twin_head {
    input intake_camshaft;
    input exhaust_camshaft;
    input chamber_volume: 46 * units.cc;
    input intake_runner_volume: 550 * units.cc;
    input intake_runner_cross_section_area: 5.5 * units.cm2;
    input exhaust_runner_volume: 450 * units.cc;
    input exhaust_runner_cross_section_area: 40 * units.cm2;

    input flow_attenuation: 5.0;
    input lift_scale: 2.6;
    input flip_display: false;
    alias output __out: head;

    function intake_flow(30 * units.thou)
    intake_flow
        .add_flow_sample(0 * lift_scale, 0 * flow_attenuation)
        .add_flow_sample(10 * lift_scale, 5.0 * flow_attenuation)
        .add_flow_sample(20 * lift_scale, 9.7 * flow_attenuation)
        .add_flow_sample(30 * lift_scale, 14.0 * flow_attenuation)
        .add_flow_sample(40 * lift_scale, 18.0 * flow_attenuation)
        .add_flow_sample(50 * lift_scale, 21.6 * flow_attenuation)
        .add_flow_sample(60 * lift_scale, 25.2 * flow_attenuation)
        .add_flow_sample(70 * lift_scale, 28.6 * flow_attenuation)
        .add_flow_sample(80 * lift_scale, 31.7 * flow_attenuation)
        .add_flow_sample(90 * lift_scale, 34.6 * flow_attenuation)
        .add_flow_sample(100 * lift_scale, 37.3 * flow_attenuation)
        .add_flow_sample(110 * lift_scale, 39.8 * flow_attenuation)
        .add_flow_sample(120 * lift_scale, 42.0 * flow_attenuation)
        .add_flow_sample(130 * lift_scale, 44.0 * flow_attenuation)
        .add_flow_sample(140 * lift_scale, 46.5 * flow_attenuation)
        .add_flow_sample(150 * lift_scale, 49.0 * flow_attenuation)

    function exhaust_flow(30 * units.thou)
    exhaust_flow
        .add_flow_sample(0 * lift_scale, 0 * flow_attenuation)
        .add_flow_sample(10 * lift_scale, 4.2 * flow_attenuation)
        .add_flow_sample(20 * lift_scale, 8.3 * flow_attenuation)
        .add_flow_sample(30 * lift_scale, 11.9 * flow_attenuation)
        .add_flow_sample(40 * lift_scale, 15.3 * flow_attenuation)
        .add_flow_sample(50 * lift_scale, 18.4 * flow_attenuation)
        .add_flow_sample(60 * lift_scale, 21.4 * flow_attenuation)
        .add_flow_sample(70 * lift_scale, 24.3 * flow_attenuation)
        .add_flow_sample(80 * lift_scale, 26.9 * flow_attenuation)
        .add_flow_sample(90 * lift_scale, 29.4 * flow_attenuation)
        .add_flow_sample(100 * lift_scale, 31.6 * flow_attenuation)
        .add_flow_sample(110 * lift_scale, 33.7 * flow_attenuation)
        .add_flow_sample(120 * lift_scale, 35.5 * flow_attenuation)
        .add_flow_sample(130 * lift_scale, 37.8 * flow_attenuation)

    generic_cylinder_head head(
        chamber_volume: chamber_volume,
        intake_runner_volume: intake_runner_volume,
        intake_runner_cross_section_area: intake_runner_cross_section_area,
        exhaust_runner_volume: exhaust_runner_volume,
        exhaust_runner_cross_section_area: exhaust_runner_cross_section_area,

        intake_port_flow: intake_flow,
        exhaust_port_flow: exhaust_flow,
        valvetrain: standard_valvetrain(
            intake_camshaft: intake_camshaft,
            exhaust_camshaft: exhaust_camshaft
        ),
        flip_display: flip_display
    )
}

// One-lobe camshaft pair; lobe offset = that cylinder's firing angle.
private node twin_cam {
    input intake_lobe_profile;
    input exhaust_lobe_profile;
    input fire_angle: 0 * units.deg;
    input intake_lobe_center: 108 * units.deg;
    input exhaust_lobe_center: 112 * units.deg;
    input base_radius: 1.0 * units.inch;

    output intake_cam_0: _intake_cam_0;
    output exhaust_cam_0: _exhaust_cam_0;

    camshaft_parameters params (
        advance: 0 * units.deg,
        base_radius: base_radius
    )

    camshaft _intake_cam_0(params, lobe_profile: intake_lobe_profile)
    camshaft _exhaust_cam_0(params, lobe_profile: exhaust_lobe_profile)

    label rot360(360 * units.deg)

    _exhaust_cam_0
        .add_lobe(rot360 - exhaust_lobe_center + fire_angle)
    _intake_cam_0
        .add_lobe(rot360 + intake_lobe_center + fire_angle)
}

'''

DEFAULTS = {
    "name": "Forge Engine",
    "layout": "inline",          # inline | v | flat | radial
    "cylinders": 4,              # total cylinders
    "rows": 1,                   # radial rows (1 or 2)
    "v_angle": 60,               # included V angle for layout v
    "displacement_l": None,      # total displacement; used if bore is not given
    "bore_mm": None,             # number | list per cylinder | {"start_mm":..,"step_mm":..,"invert_second":bool}
    "stroke_mm": None,           # default: square engine, or 90 mm for radials
    "rod_mm": None,              # default 1.7 x stroke
    "compression_ratio": 10.5,
    "firing": {"mode": "even"},  # even | pairs | groups | order | angles
    "bank_sign": 1,              # pin = fire + bank_sign*bank (flip to -1 if a bank fires off TDC)
    "redline_rpm": 8000,
    "idle_plate": 0.9985,
    "cam": {"intake_duration": 250, "exhaust_duration": 240, "intake_center": 108, "exhaust_center": 112,
            "intake_lift_thou": 400, "exhaust_lift_thou": 370},
    "exhaust": "single",         # single | per_bank | per_cylinder
    "intake_runner_ratio": 1.1,  # runner volume = ratio x swept volume
    "exhaust_runner_ratio": 0.9,
    "top_speed_mph": None,
    "third_gear_mph": None,
    "gears": 6,
    "tyre_radius_m": None,
    "vehicle_mass_kg": None,
    "sim_frequency": 5000,
    "port_flow_scale": 1.0,
    "intake_flow_scale": 1.0,
    "exhaust_flow_scale": 1.0,
    "plenum_scale": 1.0,
    "clutch_scale": 1.0,
}

STD_ORDERS = {3: [1, 2, 3], 4: [1, 3, 4, 2], 5: [1, 2, 4, 5, 3], 6: [1, 5, 3, 6, 2, 4], 8: [1, 6, 2, 5, 8, 3, 7, 4]}


def fnum(x):
    s = ("%.6f" % x).rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


# ---------------------------------------------------------------- layout & geometry
def merge(spec):
    s = json.loads(json.dumps(DEFAULTS))
    for k, v in spec.items():
        if isinstance(v, dict) and isinstance(s.get(k), dict):
            s[k].update(v)
        else:
            s[k] = v
    return s


def groups_of(spec):
    """Return list of physical groups (banks/rows) as lists of cylinder numbers, plus bank angle per cylinder."""
    n = spec["cylinders"]
    lay = spec["layout"]
    ang = {}
    grp = []
    if lay == "inline":
        grp = [list(range(1, n + 1))]
        for c in grp[0]:
            ang[c] = 0.0
    elif lay in ("v", "flat"):
        half = n // 2
        a = spec["v_angle"] / 2.0 if lay == "v" else 90.0
        grp = [list(range(1, half + 1)), list(range(half + 1, n + 1))]
        for c in grp[0]:
            ang[c] = a
        for c in grp[1]:
            ang[c] = -a
    elif lay == "radial":
        rows = spec["rows"]
        m = n // rows
        for r in range(rows):
            g = list(range(r * m + 1, (r + 1) * m + 1))
            grp.append(g)
            for k, c in enumerate(g):
                ang[c] = -(360.0 * k / m) - (180.0 / m) * r
    else:
        raise SystemExit("unknown layout %r" % lay)
    return grp, ang


def stroke_bore(spec, grp):
    n = spec["cylinders"]
    b = spec["bore_mm"]
    st = spec["stroke_mm"]
    if st is None:
        st = 90.0 if spec["layout"] == "radial" else None
    bores = {}
    if isinstance(b, (int, float)):
        for c in range(1, n + 1):
            bores[c] = float(b)
        if st is None:
            st = float(b)
    elif isinstance(b, list):
        for c, v in enumerate(b, 1):
            bores[c] = float(v)
        if st is None:
            st = float(b[0])
    elif isinstance(b, dict):
        if st is None:
            st = 90.0
        for gi, g in enumerate(grp):
            for k, c in enumerate(g):
                kk = (len(g) - 1 - k) if (b.get("invert_second") and gi % 2 == 1) else k
                bores[c] = b["start_mm"] + b["step_mm"] * kk
    else:
        disp = spec["displacement_l"] or 2.0
        per = disp * 1e6 / n  # mm3
        if st is None:
            st = (4 * per / math.pi) ** (1 / 3.0)
        d = math.sqrt(4 * per / (math.pi * st))
        for c in range(1, n + 1):
            bores[c] = d
    return float(st), bores


def solve_taper(spec, grp, st):
    """If bore_mm is a taper dict with 'start_mm' null, fit start so total displacement matches."""
    b = spec["bore_mm"]
    if isinstance(b, dict) and b.get("start_mm") is None:
        target = (spec["displacement_l"] or 3.0) * 1e6
        lo, hi = 20.0, 300.0
        for _ in range(60):
            mid = (lo + hi) / 2
            tot = 0
            for gi, g in enumerate(grp):
                for k, c in enumerate(g):
                    d = mid + b["step_mm"] * k
                    tot += math.pi / 4 * d * d * st
            if tot > target:
                hi = mid
            else:
                lo = mid
        b["start_mm"] = round((lo + hi) / 2, 1)


def firing_angles(spec, grp, ang):
    n = spec["cylinders"]
    f = spec["firing"]
    mode = f.get("mode", "even")
    fire = {}
    if mode == "angles":
        for k, v in f["angles"].items():
            fire[int(k)] = float(v) % 720
        return fire
    if spec["layout"] == "radial" and mode in ("even", "pairs") and "order" not in f:
        rows = spec["rows"]
        m = n // rows
        step = 360.0 / m
        for r, g in enumerate(grp):
            for k, c in enumerate(g):
                if r == 0:
                    fire[c] = step * k + 360 * (k % 2)
                else:
                    fire[c] = step * (k + 1) + 360 * (k % 2)
        for c in fire:
            fire[c] %= 720
        if mode == "even":
            return fire
        order = sorted(fire, key=lambda c: fire[c])
    else:
        if "order" in f:
            order = f["order"]
        elif spec["layout"] == "inline" and n in STD_ORDERS:
            order = STD_ORDERS[n]
        elif spec["layout"] in ("v", "flat"):
            h = n // 2
            order = []
            for i in range(h):
                order += [i + 1, h + i + 1]
        else:
            order = list(range(1, n + 1))
    if mode == "groups":
        gl = f["groups"]
    elif mode == "pairs":
        gl = [order[i:i + 2] for i in range(0, len(order), 2)]
    else:
        gl = [[c] for c in order]
    t = 720.0 / len(gl)
    for i, g in enumerate(gl):
        for c in g:
            fire[c] = (t * i) % 720
    missing = [c for c in range(1, n + 1) if c not in fire]
    if missing:
        raise SystemExit("firing spec leaves cylinders unassigned: %s" % missing)
    return fire


# ---------------------------------------------------------------- gearbox
def gearing(spec, litres):
    bike = litres < 2.0
    r = spec["tyre_radius_m"] or (0.31 if bike else 0.33)
    top = spec["top_speed_mph"] or (180 if bike else 160)
    third = spec["third_gear_mph"] or (100 if bike else 90)
    red = spec["redline_rpm"]
    total_top = red / 60.0 * 2 * math.pi * r / (top / 2.23694)  # final drive with 1.0 top gear
    g_top = 1.0
    g3 = top / third
    ng = spec["gears"]
    g1 = g3 * 1.75
    gears = []
    if ng == 6:
        g2 = math.sqrt(g1 * g3)
        g4 = g3 * (g_top / g3) ** (1 / 3.0)
        g5 = g3 * (g_top / g3) ** (2 / 3.0)
        gears = [g1, g2, g3, g4, g5, g_top]
    else:
        gears = [g1 * (g_top / g1) ** (i / (ng - 1.0)) for i in range(ng)]
    return [round(g, 2) for g in gears], round(total_top, 2), r, bike


# ---------------------------------------------------------------- emitter
def build(spec_in):
    spec = merge(spec_in)
    n = spec["cylinders"]
    grp, ang = groups_of(spec)
    st, _ = None, None
    st0 = spec["stroke_mm"] or (90.0 if spec["layout"] == "radial" else None)
    if isinstance(spec["bore_mm"], dict):
        solve_taper(spec, grp, st0 or 90.0)
    st, bores = stroke_bore(spec, grp)
    rod = spec["rod_mm"] or round(1.7 * st, 1)
    fire = firing_angles(spec, grp, ang)
    sgn = spec["bank_sign"]
    pin = {c: (fire[c] + sgn * ang[c]) % 360 for c in range(1, n + 1)}
    pin = {c: (0.0 if abs(p - 360) < 1e-6 else p) for c, p in pin.items()}
    swept = {c: math.pi / 4 * (bores[c] / 10) ** 2 * (st / 10) for c in bores}  # cc
    total_l = sum(swept.values()) / 1000.0
    pins = []
    for c in range(1, n + 1):
        if not any(abs(pin[c] - p) < 1e-4 for p in pins):
            pins.append(pin[c])
    pname = {}
    for c in range(1, n + 1):
        for i, p in enumerate(pins):
            if abs(pin[c] - p) < 1e-4:
                pname[c] = "rj%d" % (i + 1)
    # banks: same angle + same bore share a bank
    banks = []
    bank_of = {}
    for c in range(1, n + 1):
        key = (round(ang[c], 4), round(bores[c], 3))
        if key not in [b[0] for b in banks]:
            banks.append((key, []))
        for b in banks:
            if b[0] == key:
                b[1].append(c)
    for i, b in enumerate(banks):
        for c in b[1]:
            bank_of[c] = i
    gears, diff, tyre, bike = gearing(spec, total_l)
    red = spec["redline_rpm"]
    cam = spec["cam"]
    L = total_l
    mass_v = spec["vehicle_mass_kg"] or (300 if bike else round(250 + 140 * L))
    top_k = int(math.ceil(red / 1000.0))
    order = sorted(range(1, n + 1), key=lambda c: (fire[c], c))

    o = []
    w = o.append
    w("// %s - generated by engine_forge. %d cylinders, layout %s, ~%.2f L, stroke %.1f mm, redline %d rpm." % (
        spec["name"], n, spec["layout"], total_l, st, red))
    w("// Firing order: %s. Journal angle = firing angle + bank angle (mod 360), so each cylinder fires at its TDC." % "-".join(
        str(c) for c in order))
    w("// Tuning numbers are scaled estimates, not measured data. Untested in engine-sim. bank_sign=%d" % sgn)
    w('import "engine_sim.mr"\n\nunits units()\nconstants constants()\nimpulse_response_library ir_lib()\nlabel cycle(2 * 360 * units.deg)\n')
    w("private node wires {")
    for c in range(1, n + 1):
        w("    output wire%d: ignition_wire();" % c)
    w("}\n")
    w(HEAD_AND_CAM)
    w("public node forge_engine {")
    w("    alias output __out: engine;\n")
    w("""    function flame_ratio(5.0)
    flame_ratio
        .add_sample(0.0, 30.0)
        .add_sample(5.0, 75.0)
        .add_sample(10.0, 12.0 * 10.0)
        .add_sample(15.0, 12.0 * 15.0)
        .add_sample(20.0, 12.0 * 20.0)
        .add_sample(25.0, 12.0 * 25.0)
        .add_sample(30.0, 12.0 * 30.0)
        .add_sample(35.0, 12.0 * 35.0)
        .add_sample(40.0, 12.0 * 40.0)
        .add_sample(45.0, 12.0 * 45.0)
""")
    w("""    engine engine(
        name: "%s",
        starter_torque: %d * units.lb_ft,
        starter_speed: 300 * units.rpm,
        redline: %d * units.rpm,
        fuel: fuel(
            turbulence_to_flame_speed_ratio: flame_ratio,
            max_turbulence_effect: 1.0,
            low_efficiency_attenuation: 1.0,
            max_dilution_effect: 1000.0,
            max_burning_efficiency: 1.0,
            burning_efficiency_randomness: 0.3
        ),
        throttle_gamma: 2.0,
        simulation_frequency: %d
    )
""" % (spec["name"], round(50 * L), red, spec["sim_frequency"]))
    w("    wires wires()\n")
    w("    label stroke(%s * units.cm)" % fnum(st / 10))
    w("    label rod_length(%s * units.cm)" % fnum(rod / 10))
    w("    label compression_height(%s * units.cm)" % fnum(round(0.38 * st / 10 + 0.0, 2) if st < 70 else 3.3))
    w("    label crank_mass(%s * units.kg)" % fnum(round(8 + 4.0 * L, 1)))
    w("    label flywheel_mass(%s * units.kg)" % fnum(round(6 + 1.2 * L, 1)))
    w("    label flywheel_radius(%s * units.cm)\n" % ("15" if L < 3 else "20"))
    w("""    label crank_moment(
        disk_moment_of_inertia(mass: crank_mass, radius: stroke)
    )
    label flywheel_moment(
        disk_moment_of_inertia(mass: flywheel_mass, radius: flywheel_radius)
    )
    label other_moment(
        disk_moment_of_inertia(mass: 1.0 * units.kg, radius: 4.0 * units.cm)
    )
""")
    w("""    crankshaft c0(
        throw: stroke / 2,
        flywheel_mass: flywheel_mass,
        mass: crank_mass,
        friction_torque: %s * units.lb_ft,
        moment_of_inertia:
            crank_moment + flywheel_moment + other_moment,
        position_x: 0.0,
        position_y: 0.0,
        tdc: 90 * units.deg
    )
""" % fnum(round(4 + 3.0 * L, 1)))
    for i, p in enumerate(pins):
        w("    rod_journal rj%d(angle: %s * units.deg)" % (i + 1, fnum(p)))
    w("    c0")
    for i in range(len(pins)):
        w("        .add_rod_journal(rj%d)" % (i + 1))
    w("""
    intake intake_a(
        plenum_volume: %s * units.L,
        plenum_cross_section_area: %d * units.cm2,
        intake_flow_rate: k_carb(%d.0),
        runner_flow_rate: k_carb(1400.0),
        runner_length: 8.0 * units.inch,
        idle_flow_rate: k_carb(0.0),
        idle_throttle_plate_position: %s,
        throttle_gamma: 2.0,
        velocity_decay: 0.8
    )
""" % (fnum(round(max(1.0, 2.4 * L * spec["plenum_scale"]), 1)), round(32 * L * spec["plenum_scale"]), round(800 * L * spec["intake_flow_scale"]), fnum(spec["idle_plate"])))
    ex_mode = spec["exhaust"]
    if ex_mode == "single":
        ex_names = {c: "exhaust_main" for c in range(1, n + 1)}
        ex_list = [("exhaust_main", L)]
    elif ex_mode == "per_bank":
        ex_names = {}
        ex_list = []
        for gi, g in enumerate(grp):
            nm = "exhaust_%d" % (gi + 1)
            ex_list.append((nm, L * len(g) / n))
            for c in g:
                ex_names[c] = nm
    else:
        ex_names = {c: "exhaust_%d" % c for c in range(1, n + 1)}
        ex_list = [("exhaust_%d" % c, swept[c] / 1000.0) for c in range(1, n + 1)]
    for nm, lit in ex_list:
        w("""    exhaust_system_parameters es_params_%s(
        outlet_flow_rate: k_carb(%d.0),
        primary_tube_length: 10.0 * units.inch,
        primary_flow_rate: k_carb(3000.0),
        velocity_decay: 1.2
    )
    exhaust_system %s(
        es_params_%s,
        length: 20 * units.inch,
        collector_cross_section_area: %s * units.cm2,
        audio_volume: 1.0,
        impulse_response: ir_lib.mild_exhaust_0
    )
""" % (nm, max(600, round(1440 * lit * spec["exhaust_flow_scale"])), nm, nm, fnum(max(6.0, round(12 * lit * spec["exhaust_flow_scale"], 1)))))
    w("""    harmonic_cam_lobe intake_lobe(
        duration_at_50_thou: %s * units.deg,
        gamma: 1.1,
        lift: %s * units.thou,
        steps: 256
    )

    harmonic_cam_lobe exhaust_lobe(
        duration_at_50_thou: %s * units.deg,
        gamma: 1.1,
        lift: %s * units.thou,
        steps: 256
    )
""" % (cam["intake_duration"], cam["intake_lift_thou"], cam["exhaust_duration"], cam["exhaust_lift_thou"]))
    # per-bank parameter blocks
    for i, (key, cyls) in enumerate(banks):
        w("    cylinder_bank_parameters bank_params_%d(\n        bore: %s * units.cm,\n        deck_height: stroke / 2 + rod_length + compression_height\n    )" % (
            i + 1, fnum(round(bores[cyls[0]] / 10, 3))))
        w("    cylinder_bank b%d(bank_params_%d, angle: %s * units.deg)\n" % (i + 1, i + 1, fnum(key[0])))
    for c in range(1, n + 1):
        sc = (bores[c] / 100.0) ** 2
        sw = swept[c]
        chamber = sw / (spec["compression_ratio"] - 1)
        w("    // ---- Cylinder %d: bore %.1f mm, ~%d cc, bank angle %s, fires @%s ----" % (c, bores[c], round(sw), fnum(ang[c]), fnum(fire[c])))
        w("    piston_parameters piston_%d(\n        mass: %.3f * units.kg,\n        compression_height: compression_height,\n        wrist_pin_position: 0.0,\n        displacement: 0.0\n    )" % (c, 0.9 * sc * (st / 90.0)))
        w("    label rod_mass_%d(%.3f * units.kg)" % (c, 1.0 * sc))
        w("    connecting_rod_parameters cr_%d(\n        mass: rod_mass_%d,\n        moment_of_inertia: rod_moment_of_inertia(\n            mass: rod_mass_%d,\n            length: rod_length\n        ),\n        center_of_mass: 0.0,\n        length: rod_length\n    )" % (c, c, c))
        w("    b%d\n        .add_cylinder(\n            piston: piston(piston_%d, blowby: k_28inH2O(0.03)),\n            connecting_rod: connecting_rod(cr_%d),\n            rod_journal: %s,\n            intake: intake_a,\n            exhaust_system: %s,\n            ignition_wire: wires.wire%d,\n            sound_attenuation: 1.0,\n            primary_length: 0.0 * units.inch\n        )" % (
            bank_of[c] + 1, c, c, pname[c], ex_names[c], c))
        w("    twin_cam cam_%d(\n        intake_lobe_profile: intake_lobe,\n        exhaust_lobe_profile: exhaust_lobe,\n        intake_lobe_center: %s * units.deg,\n        exhaust_lobe_center: %s * units.deg,\n        fire_angle: %s * units.deg\n    )" % (
            c, cam["intake_center"], cam["exhaust_center"], fnum(fire[c])))
        w("    b%d\n        .set_cylinder_head(\n            twin_head(\n                intake_camshaft: cam_%d.intake_cam_0,\n                exhaust_camshaft: cam_%d.exhaust_cam_0,\n                chamber_volume: %.1f * units.cc,\n                intake_runner_volume: %d * units.cc,\n                exhaust_runner_volume: %d * units.cc,\n                flow_attenuation: %.2f)\n        )\n" % (
            bank_of[c] + 1, c, c, chamber, round(spec["intake_runner_ratio"] * sw), round(spec["exhaust_runner_ratio"] * sw), sw / 100.0 * spec["port_flow_scale"]))
    for i in range(len(banks)):
        w("    engine\n        .add_cylinder_bank(b%d)" % (i + 1))
    w("\n    engine.add_crankshaft(c0)\n")
    w("    function timing_curve(1000 * units.rpm)\n    timing_curve")
    for k in range(0, max(top_k, 8) + 2):
        adv = {0: 12, 1: 18, 2: 24, 3: 28, 4: 32, 5: 34, 6: 36, 7: 37}.get(k, 38)
        w("        .add_sample(%d * units.rpm, %d.0 * units.deg)" % (k * 1000, adv))
    w("""
    ignition_module ignition_module(
        timing_curve: timing_curve,
        rev_limit: %d * units.rpm,
        limiter_duration: 0.1)

    ignition_module""" % red)
    for c in order:
        w("            .connect_wire(wires.wire%d, %s * units.deg)" % (c, fnum(fire[c])))
    w("\n    engine.add_ignition_module(ignition_module)\n}\n")
    w("""private node forge_car {
    alias output __out:
        vehicle(
            mass: %d * units.kg,
            drag_coefficient: %s,
            cross_sectional_area: %d * units.cm2,
            diff_ratio: %s,
            tire_radius: %d * units.cm,
            rolling_resistance: %d * units.N
        );
}
""" % (mass_v, "0.6" if bike else "0.4", 7000 if bike else 20000, fnum(diff), round(tyre * 100), 40 if bike else 150 + round(L * 20)))
    w("private node forge_transmission {\n    alias output __out:\n        transmission(\n            max_clutch_torque: %d * units.lb_ft\n        )\n%s;\n}\n" % (
        round(150 * L * spec["clutch_scale"]), "\n".join("        .add_gear(%s)" % fnum(g) for g in gears)))
    w("public node main {\n    run(\n        engine: forge_engine(),\n        transmission: forge_transmission(),\n        vehicle: forge_car()\n    )\n}\n\nmain()\n")
    text = "\n".join(o)
    info = dict(spec=spec, total_l=total_l, bores=bores, fire=fire, pins=pins, order=order, gears=gears, diff=diff, stroke=st)
    return text, info


# ---------------------------------------------------------------- verification
def verify(text, tol=1e-3):
    """Re-parse a generated file and check firing/journal/cam/wire consistency. Returns list of problems."""
    probs = []
    jr = {m.group(1): float(m.group(2)) for m in re.finditer(r"rod_journal (rj\w+)\(angle: ([-\d.]+)", text)}
    blocks = re.split(r"// ---- Cylinder (\d+):", text)[1:]
    cyl = {}
    for i in range(0, len(blocks), 2):
        n = int(blocks[i])
        b = blocks[i + 1]
        bank = re.search(r"bank angle ([-\d.]+)", b)
        fire = re.search(r"fires @([-\d.]+)", b)
        rj = re.search(r"rod_journal: (rj\w+)", b)
        cam = re.search(r"fire_angle: ([-\d.]+)", b)
        cyl[n] = dict(bank=float(bank.group(1)), fire=float(fire.group(1)), rj=rj.group(1), cam=float(cam.group(1)))
    wires = {}
    for m in re.finditer(r"\.connect_wire\(wires\.wire(\d+), ([-\d.]+) \* units\.deg\)", text):
        wires.setdefault(int(m.group(1)), []).append(float(m.group(2)))
    sign = 1
    mm = re.search(r"bank_sign=(-?1)", text)
    if mm:
        sign = int(mm.group(1))
    for n, c in sorted(cyl.items()):
        j = jr.get(c["rj"])
        if j is None:
            probs.append("cyl %d uses unknown journal %s" % (n, c["rj"]))
            continue
        d = ((c["fire"] + sign * c["bank"]) - j) % 360
        if min(d, 360 - d) > tol:
            probs.append("cyl %d: journal %.4f != fire+bank %.4f" % (n, j, (c["fire"] + sign * c["bank"]) % 360))
        if abs(c["cam"] - c["fire"]) > tol:
            probs.append("cyl %d: cam lobe offset %.4f != firing angle %.4f" % (n, c["cam"], c["fire"]))
        if n not in wires or len(wires[n]) != 1:
            probs.append("cyl %d: wire connected %d times" % (n, len(wires.get(n, []))))
        elif abs(wires[n][0] - c["fire"]) > tol:
            probs.append("cyl %d: ignition angle %.4f != firing angle %.4f" % (n, wires[n][0], c["fire"]))
    extra = set(wires) - set(cyl)
    if extra:
        probs.append("wires with no cylinder: %s" % sorted(extra))
    return probs


# ---------------------------------------------------------------- prompt parsing
def parse_prompt(p):
    t = p.lower()
    spec = {}
    notes = []
    m = re.search(r"twin[- ]?row\s+radial\s*(\d+)?|(\d+)\s*cyl(?:inder)?\s+twin[- ]?row\s+radial", t)
    if m:
        spec["layout"], spec["rows"] = "radial", 2
        num = m.group(1) or m.group(2)
        spec["cylinders"] = int(num) if num else 14
    else:
        m = re.search(r"radial\s*-?(\d+)|(\d+)\s*cyl(?:inder)?\s+radial", t)
        if m:
            spec["layout"], spec["rows"], spec["cylinders"] = "radial", 1, int(m.group(1) or m.group(2))
        elif "radial" in t:
            spec["layout"], spec["rows"], spec["cylinders"] = "radial", 1, 7
        else:
            m = re.search(r"\bv\s*-?(\d+)\b", t)
            f = re.search(r"\b(?:flat|boxer)\s*-?(\d+)\b", t)
            i = re.search(r"\b(?:inline|straight|i)\s*-?(\d+)\b", t)
            if m:
                spec["layout"], spec["cylinders"] = "v", int(m.group(1))
            elif f:
                spec["layout"], spec["cylinders"] = "flat", int(f.group(1))
            elif i:
                spec["layout"], spec["cylinders"] = "inline", int(i.group(1))
            else:
                notes.append("no layout found; defaulting to inline 4")
    va = re.search(r"(\d+)\s*(?:deg|°|degree)[a-z ]{0,3}\bv\b|v[- ]angle\s*(?:of\s*)?(\d+)", t)
    if va:
        spec["v_angle"] = int(va.group(1) or va.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:l\b|litre|liter)", t)
    if m:
        spec["displacement_l"] = float(m.group(1))
    m = re.search(r"(\d{3,5})\s*cc", t)
    if m:
        spec["displacement_l"] = int(m.group(1)) / 1000.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*k\s*rpm|(\d{4,6})\s*rpm", t)
    if m:
        spec["redline_rpm"] = int(float(m.group(1)) * 1000) if m.group(1) else int(m.group(2))
    m = re.search(r"bore\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*mm", t)
    if m:
        spec["bore_mm"] = float(m.group(1))
    m = re.search(r"stroke\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*mm", t)
    if m:
        spec["stroke_mm"] = float(m.group(1))
    if re.search(r"taper|each cylinder (?:gets )?smaller|sequentially smaller|decreasing", t):
        step = -6.0
        sm = re.search(r"(\d+(?:\.\d+)?)\s*mm\s*(?:steps?|taper)", t)
        if sm:
            step = -float(sm.group(1))
        spec["bore_mm"] = {"start_mm": spec.get("bore_mm") if isinstance(spec.get("bore_mm"), float) else None,
                           "step_mm": step, "invert_second": bool(re.search(r"invert|opposite|reverse", t))}
        if spec["bore_mm"]["start_mm"] is None and "displacement_l" not in spec:
            spec["displacement_l"] = 3.4
            notes.append("taper without size: assuming 3.4 L total")
    if re.search(r"\bpairs?\b", t):
        spec["firing"] = {"mode": "pairs"}
    m = re.search(r"groups? of (\d+)", t)
    if m and "firing" not in spec:
        gsz = int(m.group(1))
        spec["firing"] = {"mode": "groups_of", "size": gsz}
    m = re.search(r"firing order\s*[:=]?\s*([\d\s\-,]+)", t)
    if m:
        order = [int(x) for x in re.findall(r"\d+", m.group(1))]
        if order:
            spec["firing"] = {"mode": "even", "order": order}
    m = re.search(r"(\d+(?:\.\d+)?)\s*mph", t)
    if m:
        spec["top_speed_mph"] = float(m.group(1))
    m = re.search(r"idle\s*(?:plate|throttle)?\s*(?:of|at|=)?\s*(0\.\d+)", t)
    if m:
        spec["idle_plate"] = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*:\s*1", t)
    if m:
        spec["compression_ratio"] = float(m.group(1))
    if re.search(r"per[- ]cylinder exhaust|individual exhaust", t):
        spec["exhaust"] = "per_cylinder"
    elif re.search(r"exhaust per bank|separate banks? exhaust", t):
        spec["exhaust"] = "per_bank"
    spec["name"] = (p.strip()[:48] or "Forge Engine")
    return spec, notes


def finalize_groups_of(spec):
    f = spec.get("firing", {})
    if f.get("mode") == "groups_of":
        n = spec.get("cylinders", 4)
        sz = f["size"]
        order = list(range(1, n + 1))
        spec["firing"] = {"mode": "groups", "groups": [order[i:i + sz] for i in range(0, n, sz)]}


def spec_from_llm(prompt):
    import urllib.request
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("--llm needs ANTHROPIC_API_KEY in the environment")
    system = ("Convert the user's engine description into ONE JSON object matching this schema and nothing else: "
              + json.dumps(DEFAULTS) + ". Only set fields the user specified or implied; omit the rest. "
              "firing.mode is one of even|pairs|groups|order|angles; for groups give firing.groups as lists of cylinder numbers; "
              "bore_mm may be a number, a list, or {start_mm, step_mm, invert_second}.")
    body = json.dumps({"model": os.environ.get("ENGINE_FORGE_MODEL", "claude-sonnet-5-5"), "max_tokens": 2000,
                       "system": system, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", body,
                                 {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    out = json.load(urllib.request.urlopen(req))
    txt = out["content"][0]["text"]
    return json.loads(txt[txt.index("{"): txt.rindex("}") + 1])


LAST = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".forge_last")


def ask_correction(prompt, meaning):
    import forge_brain as fb
    print("Was that the right reading? [Y/n] ", end="")
    if input().strip().lower() in ("", "y", "yes"):
        return
    labels = dict(meaning["labels"])
    for h, choices in fb.HEADS.items():
        print("%s [%s] (enter keeps %s): " % (h, "/".join(choices), labels[h]), end="")
        v = input().strip()
        if v in choices:
            labels[h] = v
    fb.teach(prompt, labels)
    print("Learned. The brain was retrained with your correction.")


def llm_labels(prompt):
    """Ask Claude (if ANTHROPIC_API_KEY is set) to label a prompt for the brain; used as a teacher on unfamiliar wording."""
    import urllib.request
    import forge_brain as fb
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    system = ("Label an engine-design request. Reply with ONE JSON object and nothing else, with these keys and allowed values: "
              + json.dumps(fb.HEADS) + ". Use 'none'/'no' when the request does not mention that aspect. Edits to an existing engine "
              "have intent 'edit' and a target; 'dir' for idle means the resulting idle RPM (up/down).")
    body = json.dumps({"model": os.environ.get("ENGINE_FORGE_MODEL", "claude-sonnet-5-5"), "max_tokens": 400, "system": system,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", body,
                                 {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        txt = json.load(urllib.request.urlopen(req, timeout=30))["content"][0]["text"]
        lab = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
    except Exception as e:  # network/format problems: fall back to the brain alone
        print("note: LLM teacher unavailable (%s)" % e)
        return None
    return {h: lab[h] for h in fb.HEADS if lab.get(h) in fb.HEADS[h]}


def main():
    ap = argparse.ArgumentParser(description="Generate engine-sim .mr engines from a prompt or JSON spec; learns what your prompts mean")
    ap.add_argument("--prompt")
    ap.add_argument("--spec")
    ap.add_argument("--edit", help="spec json to modify (default: the last engine you generated)")
    ap.add_argument("--rules", action="store_true", help="use the plain keyword parser instead of the neural network")
    ap.add_argument("--llm", action="store_true", help="use Claude (ANTHROPIC_API_KEY) to turn the prompt into a spec")
    ap.add_argument("--teach", action="store_true", help="ask whether the reading was right and learn from corrections")
    ap.add_argument("--correct", help='JSON of correct labels for --prompt, e.g. \'{"layout":"radial2","taper":"yes"}\'; teaches the brain')
    ap.add_argument("--explain", action="store_true", help="show how the brain read the prompt")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("-o", "--out", default="engine.mr")
    ap.add_argument("--example", action="store_true")
    ap.add_argument("--verify")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if a.example:
        print(json.dumps(DEFAULTS, indent=2))
        return
    if a.verify:
        probs = verify(open(a.verify).read())
        print("OK: all cylinders consistent" if not probs else "\n".join(probs))
        sys.exit(1 if probs else 0)
    if a.retrain:
        import forge_brain as fb
        fb.Brain().train(verbose=True)
        print("retrained")
        if not a.prompt:
            return
    notes, changes, meaning = [], [], None
    if a.spec:
        spec = json.load(open(a.spec))
    elif a.prompt:
        if a.llm:
            spec = spec_from_llm(a.prompt)
        elif a.rules:
            spec, notes = parse_prompt(a.prompt)
        else:
            import forge_brain as fb
            brain = fb.Brain()
            meaning = fb.interpret(a.prompt, brain)
            if a.correct:
                lab = dict(meaning["labels"])
                lab.update(json.loads(a.correct))
                fb.teach(a.prompt, lab)
                meaning = fb.interpret(a.prompt, fb.Brain())
                print("learned your correction and retrained")
            lab = meaning["labels"]
            weak = [h for h in ("intent", "layout", "target") if meaning["conf"][h] < 0.6]
            if weak and os.environ.get("ANTHROPIC_API_KEY"):
                lab2 = llm_labels(a.prompt)
                if lab2:
                    fb.teach(a.prompt, {**meaning["labels"], **lab2})
                    meaning = fb.interpret(a.prompt, fb.Brain())
                    lab = meaning["labels"]
                    print("unfamiliar wording: asked Claude, and the brain learned from its answer")
                    weak = [h for h in ("intent", "layout", "target") if meaning["conf"][h] < 0.6]
            if weak:
                notes.append("low confidence on %s; check --explain and use --correct / --teach if wrong" % ", ".join(weak))
            if a.explain or weak:
                print("reading:", {k: v for k, v in lab.items() if v not in ("none", "no")}, meaning["numbers"], "confidence", meaning["conf"])
            if lab["intent"] == "edit":
                src = a.edit or (open(LAST).read().strip() if os.path.exists(LAST) else None)
                if not src or not os.path.exists(src):
                    raise SystemExit("that sounds like an edit but there is no engine to edit yet; generate one first or pass --edit spec.json")
                spec = merge(json.load(open(src)))
                changes = fb.apply_edit(spec, meaning)
            else:
                spec = fb.spec_from_meaning(a.prompt, meaning)
            if a.teach and sys.stdin.isatty():
                pass
    else:
        ap.error("give --prompt or --spec")
    finalize_groups_of(spec)
    text, info = build(spec)
    probs = verify(text)
    if probs:
        print("INTERNAL CONSISTENCY PROBLEMS:\n" + "\n".join(probs))
        sys.exit(2)
    open(a.out, "w").write(text)
    sp = os.path.abspath(a.out) + ".spec.json"
    json.dump(info["spec"], open(sp, "w"), indent=1)
    open(LAST, "w").write(sp)
    print("wrote %s: %d cylinders, %.2f L, firing %s" % (a.out, info["spec"]["cylinders"], info["total_l"], "-".join(map(str, info["order"]))))
    print("gears %s, final drive %s, redline %d" % (info["gears"], info["diff"], info["spec"]["redline_rpm"]))
    for c in changes:
        print("changed:", c)
    for n in notes:
        print("note:", n)
    if a.teach and meaning and sys.stdin.isatty():
        ask_correction(a.prompt, meaning)


if __name__ == "__main__":
    main()
