#!/usr/bin/env python3
"""Verifica dell'efficacia della rampa di spunto sui nuovi log.

Uso:  python3 check_spool.py last_run_plots/flight_*.csv

Le due metriche che decidono se la rampa ha funzionato, entrambe misurate sui
log dell'11/9 come baseline:

  sat60k    campioni con cmd >= 59999 durante rising.
            Baseline: 0-22 per run (i tre run peggiori: 10, 14, 22).
            Atteso dopo la rampa: 0.

  rimbalzo  vbat a 2 s meno vbat a 0.5 s dall'inizio del rising.
            Positivo = la tensione era affondata SOTTO il proprio regime e sta
            risalendo, cioe' c'e' stato un picco di corrente transitorio.
            Baseline: +0.058 / +0.121 / +0.227 V nei tre run saturati,
            circa -0.04 V nei run sani.
            Atteso dopo la rampa: ~0 o leggermente negativo ovunque.
"""
import sys
import glob
import numpy as np
import pandas as pd

files = []
for a in sys.argv[1:]:
    files.extend(sorted(glob.glob(a)))
if not files:
    sys.exit("uso: python3 check_spool.py <file csv o glob>")

print("%-28s %8s %9s %9s %9s %9s" % (
    "file", "sat60k", "v@0.5s", "v@2s", "rimbalzo", "esito"))
for f in files:
    d = pd.read_csv(f)
    ris = d[d.state == "rising"]
    if ris.empty:
        print("%-28s  (nessun campione rising)" % f.split("/")[-1])
        continue
    t0 = ris.t.iloc[0]

    def at(dt):
        s = ris[ris.t <= t0 + dt]
        return s.vbat.iloc[-1] if len(s) else np.nan

    v05, v2 = at(0.5), at(2.0)
    sat = int((ris.cmd >= 59999).sum())
    reb = v2 - v05
    ok = "OK" if (sat == 0 and reb < 0.02) else "DA GUARDARE"
    print("%-28s %8d %9.3f %9.3f %+9.3f %9s" % (
        f.split("/")[-1], sat, v05, v2, reb, ok))

    # il tetto ha morso davvero? (colonne presenti solo nei log nuovi)
    if "spool_ceil" in d.columns and "cmd_ctrl" in d.columns:
        m = d.spool_ceil.notna() & d.cmd_ctrl.notna()
        if m.any():
            clipped = int((d.loc[m, "cmd_ctrl"] > d.loc[m, "spool_ceil"]).sum())
            print("%-28s   tetto attivo per %d campioni, di cui %d effettivamente tagliati"
                  % ("", int(m.sum()), clipped))
    pre = d[d.state == "prespin"]
    if len(pre):
        print("%-28s   pre-spin: %.2f s, z %.3f -> %.3f m (deve restare ferma)"
              % ("", pre.t.max() - pre.t.min(), pre.z.iloc[0], pre.z.iloc[-1]))
    sp = d[d.state == "spool"]
    if len(sp):
        cmd_lift = sp.cmd.iloc[-1]
        hold = d[d.state == "hold"]
        msg = ""
        if len(hold):
            true_hov = hold[hold.t > hold.t.max() - 0.8].cmd.mean()
            msg = ("  |  hover vero in hold %.0f, scarto al distacco %+.0f"
                   % (true_hov, cmd_lift - true_hov))
        print("%-28s   spool: %.2f s, distacco a cmd %.0f%s"
              % ("", sp.t.max() - sp.t.min(), cmd_lift, msg))

    # --- stimatore dell'hover ---
    if "hover_cmd" in d.columns:
        fly = d[d.state.isin(["rising", "nav_to_wp", "hold", "landing"])]
        frozen = int(fly.hover_raw.isna().sum())
        print("%-28s   HOVER_CMD: %.0f -> %.0f (escursione %+.0f), stima congelata "
              "su %d/%d campioni in volo"
              % ("", fly.hover_cmd.iloc[0], fly.hover_cmd.iloc[-1],
                 fly.hover_cmd.iloc[-1] - fly.hover_cmd.iloc[0], frozen, len(fly)))
        lnd = d[d.state == "landing"]
        if len(lnd):
            eq = lnd[lnd.vz.abs() < 0.06]
            if len(eq) > 10:
                # in equilibrio il cmd inviato E' l'hover vero: il residuo deve
                # restare vicino a zero se lo stimatore sta inseguendo
                res = (eq.cmd - eq.hover_cmd)
                print("%-28s   residuo in equilibrio nel landing: medio %+.0f, max |%.0f|"
                      % ("", res.mean(), res.abs().max()))
        if "accz" in d.columns and fly.accz.notna().any():
            print("%-28s   acc.z in volo: mediana %.3f (atteso ~1.0), 5-95%% %.3f-%.3f"
                  % ("", fly.accz.median(), fly.accz.quantile(.05), fly.accz.quantile(.95)))