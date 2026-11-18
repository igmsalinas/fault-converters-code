#!/usr/bin/env python3
"""Fail fast unless the Jetson PMLib daemon releases reusable counters."""

import argparse
import math
import time

from benchmark_jetson_resources import PMLibPowerMeter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-dir", required=True)
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=6526)
    parser.add_argument("--device", default="Jetson-Nano")
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=1.2)
    args = parser.parse_args()

    for index in range(args.windows):
        meter = PMLibPowerMeter(
            args.client_dir,
            args.server_ip,
            args.server_port,
            args.device,
        )
        meter.start()
        time.sleep(args.seconds)
        measurement = meter.stop()
        sample_count = int(measurement["sample_count"])
        input_power = float(
            measurement["rails"]["POM_5V_IN"]["mean_w"]
        )
        if sample_count < 2 or not math.isfinite(input_power) or input_power <= 0:
            raise RuntimeError(
                "Invalid PMLib window %d: %d samples, %.6f W"
                % (index + 1, sample_count, input_power)
            )
        print(
            "PMLib window %d/%d: %d samples, POM_5V_IN %.3f W"
            % (index + 1, args.windows, sample_count, input_power)
        )


if __name__ == "__main__":
    main()
