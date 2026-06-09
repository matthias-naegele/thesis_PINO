# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modifications copyright (c) 2026 Matthias Nägele.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from physicsnemo.sym.eq.pde import PDE
from sympy import Symbol, Function, Number, sqrt


class MHD_PDE(PDE):
    """MHD PDEs using PhysicsNeMo Sym"""

    name = "MHD_PDE"

    def __init__(self, Gamma=4.0 / 3.0):
        # x, y, time
        x, y, t = Symbol("x"), Symbol("y"), Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "t": t}

        # make functions
        u1 = Function("u1")(*input_variables)
        u2 = Function("u2")(*input_variables)
        b1 = Function("b1")(*input_variables)
        b2 = Function("b2")(*input_variables)
        rho = Function("rho")(*input_variables)
        e3 = Function("e3")(*input_variables)
        p = Function("p")(*input_variables)
        # eta can later be promoted to eta(x, y) for spatially varying resistivity.
        eta = Function("eta")(*input_variables)

        # initialize constants
        Gamma = Number(Gamma)

        self.equations = {}

        # compute Lorentz factor from u
        gamma = sqrt(1 + u1**2 + u2**2)

        # enthalpy
        w = rho + (Gamma * p) / (Gamma - 1)

        # divB = 0
        self.equations["div_B"] = b1.diff(x) + b2.diff(y)

        # Faraday-Induction:
        # del_t B + rot E = 0
        self.equations["FI1"] = b1.diff(t) + e3.diff(y)
        self.equations["FI2"] = b2.diff(t) - e3.diff(x)

        # Gauss & Maxwell-Ampère & Ohm
        self.equations["MO"] = eta * (e3.diff(t) - b2.diff(x) + b1.diff(y)) + (
            gamma * e3 + u1 * b2 - u2 * b1)

        # Equation of state:

        # del_t e + div S = 0
        e = 0.5 * (e3**2 + b1**2 + b2**2) + w * gamma**2 - p
        S1 = -e3 * b2 + w * gamma * u1
        S2 = e3 * b1 + w * gamma * u2
        self.equations["ES0"] = e.diff(t) + S1.diff(x) + S2.diff(y)

        # del_t P + div PI = 0
        P1 = S1
        P2 = S2

        r = 0.5 * (e3**2 + b1**2 + b2**2) + p
        PIxx = w * u1**2 - b1**2 + r
        PIxy = w * u1 * u2 - b1 * b2 # PIxy = PIyx
        PIyy = w * u2**2 - b2**2 + r

        self.equations["ES1"] = P1.diff(t) + PIxx.diff(x) + PIxy.diff(y)
        self.equations["ES2"] = P2.diff(t) + PIyy.diff(y) + PIxy.diff(x)


        # continuity eq
        self.equations["C1"] = (rho * gamma).diff(t) + (rho * u1).diff(x) + (rho * u2).diff(y)
