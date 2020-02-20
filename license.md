# Spitfire
A Python-C++ library for building tabulated chemistry models and solving differential equations

Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).

Version 1.0

Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this
software.

Questions? Contact Mike Hansen (mahanse@sandia.gov)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that
the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

----------------------------------------------------------------

Licenses of third-party libraries may be found at the following locations. 
- NumPy: https://numpy.org/license.html 
- SciPy: https://github.com/scipy/scipy/blob/master/LICENSE.txt 
- Cantera: https://github.com/Cantera/cantera/blob/master/License.txt 
- matplotlib (also, see below): https://matplotlib.org/users/license.html
- Cython (also, see below): https://github.com/cython/cython/blob/master/LICENSE.txt
- sphinx: https://github.com/sphinx-doc/sphinx/blob/master/LICENSE
- sphinx-rtd-theme: https://pypi.python.org/pypi/sphinx_rtd_theme/
- numpydoc: https://github.com/numpy/numpydoc/blob/master/LICENSE.txt

----------------------------------------------------------------

----
Cython copyright notice
----

Copyright 2019 Stefan Behnel, Robert Bradshaw, Dag Sverre Seljebotn, Greg Ewing, William Stein, Gabriel Gellner, et al..  
All rights reserved. Licensed under the Apache License, Version 2.0, you may not use this file except in 
compliance with the Apache License.  You may obtain a copy of the Apache License at 
http://www.apache.org/licenses/LICENSE-2.0.  Unless required by applicable law or agreed to in writing, 
software distributed under the Apache License is distributed on an “AS IS” BASIS, WITHOUT WARRENTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the Apache License for the specific language 
governing permissions and limitations under the Apache License.


----
Matplotlib license agreement Copyright © 2012-2013 Matplotlib Development Team; All Rights Reserved
----

Copyright Policy John Hunter began matplotlib around 2003. Since shortly before his passing in 2012, Michael
Droettboom has been the lead maintainer of matplotlib, but, as has always been the case, matplotlib is the
work of many.

Prior to July of 2013, and the 1.3.0 release, the copyright of the source code was held by John Hunter. As
of July 2013, and the 1.3.0 release, matplotlib has moved to a shared copyright model.

matplotlib uses a shared copyright model. Each contributor maintains copyright over their contributions to
matplotlib. But, it is important to note that these contributions are typically only changes to the
repositories. Thus, the matplotlib source code, in its entirety, is not the copyright of any single person
or institution. Instead, it is the collective copyright of the entire matplotlib Development Team. If
individual contributors want to maintain a record of what changes/contributions they have specific copyright
on, they should indicate their copyright in the commit message of the change, when they commit the change to
one of the matplotlib repositories.

The Matplotlib Development Team is the set of all contributors to the matplotlib project. A full list can be
obtained from the git version control logs.

License agreement for matplotlib 3.1.1 1. This LICENSE AGREEMENT is between the Matplotlib Development Team
("MDT"), and the Individual or Organization ("Licensee") accessing and otherwise using matplotlib software
in source or binary form and its associated documentation.

2. Subject to the terms and conditions of this License Agreement, MDT hereby grants Licensee a nonexclusive,
royalty-free, world-wide license to reproduce, analyze, test, perform and/or display publicly, prepare
derivative works, distribute, and otherwise use matplotlib 3.1.1 alone or in any derivative version,
provided, however, that MDT's License Agreement and MDT's notice of copyright, i.e., "Copyright (c)
2012-2013 Matplotlib Development Team; All Rights Reserved" are retained in matplotlib 3.1.1 alone or in any
derivative version prepared by Licensee.

3. In the event Licensee prepares a derivative work that is based on or incorporates matplotlib 3.1.1 or any
part thereof, and wants to make the derivative work available to others as provided herein, then Licensee
hereby agrees to include in any such work a brief summary of the changes made to matplotlib 3.1.1.

4. MDT is making matplotlib 3.1.1 available to Licensee on an "AS IS" basis. MDT MAKES NO REPRESENTATIONS OR
WARRANTIES, EXPRESS OR IMPLIED. BY WAY OF EXAMPLE, BUT NOT LIMITATION, MDT MAKES NO AND DISCLAIMS ANY
REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF
MATPLOTLIB 3.1.1 WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

5. MDT SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF MATPLOTLIB 3.1.1 FOR ANY INCIDENTAL, SPECIAL,
OR CONSEQUENTIAL DAMAGES OR LOSS AS A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING MATPLOTLIB
3.1.1, OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material breach of its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any relationship of agency, partnership, or
joint venture between MDT and Licensee. This License Agreement does not grant permission to use MDT
trademarks or trade name in a trademark sense to endorse or promote products or services of Licensee, or any
third party.

8. By copying, installing or otherwise using matplotlib 3.1.1, Licensee agrees to be bound by the terms and
conditions of this License Agreement.

License agreement for matplotlib versions prior to 1.3.0 1. This LICENSE AGREEMENT is between John D. Hunter
("JDH"), and the Individual or Organization ("Licensee") accessing and otherwise using matplotlib software
in source or binary form and its associated documentation.

2. Subject to the terms and conditions of this License Agreement, JDH hereby grants Licensee a nonexclusive,
royalty-free, world-wide license to reproduce, analyze, test, perform and/or display publicly, prepare
derivative works, distribute, and otherwise use matplotlib 3.1.1 alone or in any derivative version,
provided, however, that JDH's License Agreement and JDH's notice of copyright, i.e., "Copyright (c)
2002-2009 John D. Hunter; All Rights Reserved" are retained in matplotlib 3.1.1 alone or in any derivative
version prepared by Licensee.

3. In the event Licensee prepares a derivative work that is based on or incorporates matplotlib 3.1.1 or any
part thereof, and wants to make the derivative work available to others as provided herein, then Licensee
hereby agrees to include in any such work a brief summary of the changes made to matplotlib 3.1.1.

4. JDH is making matplotlib 3.1.1 available to Licensee on an "AS IS" basis. JDH MAKES NO REPRESENTATIONS OR
WARRANTIES, EXPRESS OR IMPLIED. BY WAY OF EXAMPLE, BUT NOT LIMITATION, JDH MAKES NO AND DISCLAIMS ANY
REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF
MATPLOTLIB 3.1.1 WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

5. JDH SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF MATPLOTLIB 3.1.1 FOR ANY INCIDENTAL, SPECIAL,
OR CONSEQUENTIAL DAMAGES OR LOSS AS A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING MATPLOTLIB
3.1.1, OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material breach of its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any relationship of agency, partnership, or
joint venture between JDH and Licensee. This License Agreement does not grant permission to use JDH
trademarks or trade name in a trademark sense to endorse or promote products or services of Licensee, or any
third party.

8. By copying, installing or otherwise using matplotlib 3.1.1, Licensee agrees to be bound by the terms and
conditions of this License Agreement.