/* 
 * Spitfire - a Python-C++ library for building tabulated chemistry models and solving differential equations                    
 * Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
 *                      
 * You should have received a copy of the 3-clause BSD License                                        
 * along with this program.  If not, see <https://opensource.org/licenses/BSD-3-Clause>.   
 *                   
 * Questions? Contact Mike Hansen (mahanse@sandia.gov)    
 */
#include "combustion_kernels.h"
#include <map>

namespace griffon {

std::map<std::string, double> get_element_mw_map() {
  std::map<std::string, double> element_mw_map;
  element_mw_map["H"] = 1.00794;
  element_mw_map["D"] = 2.014102;
  element_mw_map["Tr"] = 3.016327;
  element_mw_map["He"] = 4.00260;
  element_mw_map["Li"] = 6.941;
  element_mw_map["Be"] = 9.012182;
  element_mw_map["B"] = 10.811;
  element_mw_map["C"] = 12.011;
  element_mw_map["N"] = 14.00674;
  element_mw_map["O"] = 15.9994;
  element_mw_map["F"] = 18.9984032;
  element_mw_map["Ne"] = 20.1797;
  element_mw_map["Na"] = 22.98977;
  element_mw_map["Mg"] = 24.3050;
  element_mw_map["Al"] = 26.98154;
  element_mw_map["Si"] = 28.0855;
  element_mw_map["P"] = 30.97376;
  element_mw_map["S"] = 32.066;
  element_mw_map["Cl"] = 35.4527;
  element_mw_map["Ar"] = 39.948;
  element_mw_map["K"] = 39.0983;
  element_mw_map["Ca"] = 40.078;
  element_mw_map["Sc"] = 44.95591;
  element_mw_map["Ti"] = 47.88;
  element_mw_map["V"] = 50.9415;
  element_mw_map["Cr"] = 51.9961;
  element_mw_map["Mn"] = 54.9381;
  element_mw_map["Fe"] = 55.847;
  element_mw_map["Co"] = 58.9332;
  element_mw_map["Ni"] = 58.69;
  element_mw_map["Cu"] = 63.546;
  element_mw_map["Zn"] = 65.38;
  element_mw_map["Ga"] = 69.723;
  element_mw_map["Ge"] = 72.61;
  element_mw_map["As"] = 74.92159;
  element_mw_map["Se"] = 78.96;
  element_mw_map["Br"] = 79.904;
  element_mw_map["Kr"] = 83.80;
  element_mw_map["Rb"] = 85.4678;
  element_mw_map["Sr"] = 87.62;
  element_mw_map["Y"] = 88.90585;
  element_mw_map["Zr"] = 91.224;
  element_mw_map["Nb"] = 92.90638;
  element_mw_map["Mo"] = 95.94;
  element_mw_map["Tc"] = 97.9072;
  element_mw_map["Ru"] = 101.07;
  element_mw_map["Rh"] = 102.9055;
  element_mw_map["Pd"] = 106.42;
  element_mw_map["Ag"] = 107.8682;
  element_mw_map["Cd"] = 112.411;
  element_mw_map["In"] = 114.82;
  element_mw_map["Sn"] = 118.710;
  element_mw_map["Sb"] = 121.75;
  element_mw_map["Te"] = 127.6;
  element_mw_map["I"] = 126.90447;
  element_mw_map["Xe"] = 131.29;
  element_mw_map["Cs"] = 132.90543;
  element_mw_map["Ba"] = 137.327;
  element_mw_map["La"] = 138.9055;
  element_mw_map["Ce"] = 140.115;
  element_mw_map["Pr"] = 140.90765;
  element_mw_map["Nd"] = 144.24;
  element_mw_map["Pm"] = 144.9127;
  element_mw_map["Sm"] = 150.36;
  element_mw_map["Eu"] = 151.965;
  element_mw_map["Gd"] = 157.25;
  element_mw_map["Tb"] = 158.92534;
  element_mw_map["Dy"] = 162.50;
  element_mw_map["Ho"] = 164.93032;
  element_mw_map["Er"] = 167.26;
  element_mw_map["Tm"] = 168.93421;
  element_mw_map["Yb"] = 173.04;
  element_mw_map["Lu"] = 174.967;
  element_mw_map["Hf"] = 178.49;
  element_mw_map["Ta"] = 180.9479;
  element_mw_map["W"] = 183.85;
  element_mw_map["Re"] = 186.207;
  element_mw_map["Os"] = 190.2;
  element_mw_map["Ir"] = 192.22;
  element_mw_map["Pt"] = 195.08;
  element_mw_map["Au"] = 196.96654;
  element_mw_map["Hg"] = 200.59;
  element_mw_map["Ti"] = 204.3833;
  element_mw_map["Pb"] = 207.2;
  element_mw_map["Bi"] = 208.98037;
  element_mw_map["Po"] = 208.9824;
  element_mw_map["At"] = 209.9871;
  element_mw_map["Rn"] = 222.0176;
  element_mw_map["Fr"] = 223.0197;
  element_mw_map["Ra"] = 226.0254;
  element_mw_map["Ac"] = 227.0279;
  element_mw_map["Th"] = 232.0381;
  element_mw_map["Pa"] = 231.03588;
  element_mw_map["U"] = 238.0508;
  element_mw_map["Np"] = 237.0482;
  element_mw_map["Pu"] = 244.0482;
  element_mw_map["E"] = 0.000545;
  return element_mw_map;
}
}
