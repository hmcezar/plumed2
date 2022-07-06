/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2017-2022 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
/*
 This class was originally written by Henrique Musseli Cezar
 and is based on the SAXS class.
*/

#include "MetainferenceBase.h"
#include "core/ActionRegister.h"
#include "core/ActionSet.h"
#include "core/GenericMolInfo.h"
#include "tools/Communicator.h"
#include "tools/Pbc.h"

#include <map>

#include <iostream>

#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif

namespace PLMD {
namespace isdb {

//+PLUMEDOC ISDB_COLVAR SANS
/*
Calculates SANS scattered intensity using the Debye equation.

Intensities are calculated for a set of scattering length set using QVALUE keywords that are numbered starting from 0.
Structure factors can be either assigned using a polynomial expansion to any order using the PARAMETERS keywords;
automatically assigned to atoms using the ATOMISTIC flag reading a PDB file, a correction for the water density is
automatically added, with water density that by default is 0.334 but that can be set otherwise using WATERDENS;
automatically assigned to Martini pseudo atoms using the MARTINI flag.
The calculated intensities can be scaled using the SCALEINT keywords. This is applied by rescaling the structure factors.
Experimental reference intensities can be added using the EXPINT keywords.
By default SAXS is calculated using Debye on CPU, by adding the GPU flag it is possible to solve the equation on a GPU
if the ARRAYFIRE libraries are installed and correctly linked.
\ref METAINFERENCE can be activated using DOSCORE and the other relevant keywords.

\par Examples
in the following example the saxs intensities for a martini model are calculated. structure factors
are obtained from the pdb file indicated in the MOLINFO.

\plumedfile
MOLINFO STRUCTURE=template.pdb

SAXS ...
LABEL=saxs
ATOMS=1-355
SCALEINT=3920000
MARTINI
QVALUE1=0.02 EXPINT1=1.0902
QVALUE2=0.05 EXPINT2=0.790632
QVALUE3=0.08 EXPINT3=0.453808
QVALUE4=0.11 EXPINT4=0.254737
QVALUE5=0.14 EXPINT5=0.154928
QVALUE6=0.17 EXPINT6=0.0921503
QVALUE7=0.2 EXPINT7=0.052633
QVALUE8=0.23 EXPINT8=0.0276557
QVALUE9=0.26 EXPINT9=0.0122775
QVALUE10=0.29 EXPINT10=0.00880634
QVALUE11=0.32 EXPINT11=0.0137301
QVALUE12=0.35 EXPINT12=0.0180036
QVALUE13=0.38 EXPINT13=0.0193374
QVALUE14=0.41 EXPINT14=0.0210131
QVALUE15=0.44 EXPINT15=0.0220506
... SAXS

PRINT ARG=(saxs\.q-.*),(saxs\.exp-.*) FILE=colvar STRIDE=1

\endplumedfile

*/
//+ENDPLUMEDOC

class SANS :
  public MetainferenceBase
{
private:
  enum { H, D, C, N, O, P, S, NTT };

  bool                       pbc;
  bool                       serial;
  double                     SL_rank = 0.;
  double                     scale_int = 1.;
  std::vector<double>        q_list;
  std::vector<double>        SL_value;

  void radial_distribution_hist();
  void calculate_cpu(std::vector<Vector> &deriv);
  void tableASL(const std::vector<AtomNumber> &atoms, std::vector<double> &SL_value);

public:
  static void registerKeywords( Keywords& keys );
  explicit SANS(const ActionOptions&);
  void calculate() override;
  void update() override;
};

PLUMED_REGISTER_ACTION(SANS,"SANS")

void SANS::registerKeywords(Keywords& keys) {
  componentsAreNotOptional(keys);
  MetainferenceBase::registerKeywords(keys);
  keys.addFlag("NOPBC",false,"ignore the periodic boundary conditions when calculating distances");
  keys.addFlag("SERIAL",false,"Perform the calculation in serial - for debug purpose");
  keys.addFlag("ATOMISTIC",false,"calculate SANS for an atomistic model");
  keys.addFlag("MARTINI",false,"calculate SANS for a Martini model");
  keys.add("atoms","ATOMS","The atoms to be included in the calculation, e.g. the whole protein.");
  keys.add("numbered","QVALUE","Selected scattering lengths in Angstrom are given as QVALUE1, QVALUE2, ... .");
  keys.add("numbered","SCATLEN","Use SCATLEN keyword like SCATLEN1, SCATLEN2. These are the scattering lengths for the \\f$i\\f$th atom/bead.");
  keys.add("numbered","EXPINT","Add an experimental value for each q value.");
  keys.add("compulsory","SCALEINT","1.0","SCALING value of the calculated data. Useful to simplify the comparison.");
  keys.addOutputComponent("q","default","the # SAXS of q");
  keys.addOutputComponent("exp","EXPINT","the # experimental intensity");
}

SANS::SANS(const ActionOptions&ao):
  PLUMED_METAINF_INIT(ao),
  pbc(true),
  serial(false)
{
  std::vector<AtomNumber> atoms;
  parseAtomList("ATOMS",atoms);
  const unsigned size = atoms.size();

  parseFlag("SERIAL",serial);

  bool nopbc=!pbc;
  parseFlag("NOPBC",nopbc);
  pbc=!nopbc;
  if(pbc)      log.printf("  using periodic boundary conditions\n");
  else         log.printf("  without periodic boundary conditions\n");

  unsigned ntarget=0;
  for(unsigned i=0;; ++i) {
    double t_list;
    if( !parseNumbered( "QVALUE", i+1, t_list) ) break;
    if(t_list<=0.) error("QVALUE cannot be less or equal to zero!\n");
    q_list.push_back(t_list);
    ntarget++;
  }
  const unsigned numq = ntarget;

  for(unsigned i=0; i<numq; i++) {
    if(q_list[i]==0.) error("it is not possible to set q=0\n");
    if(i>0&&q_list[i]<q_list[i-1]) error("QVALUE must be in ascending order");
    log.printf("  my q: %lf \n",q_list[i]);
  }

  bool atomistic=false;
  parseFlag("ATOMISTIC",atomistic);
  bool martini=false;
  parseFlag("MARTINI",martini);

  if(martini&&atomistic) error("You cannot use MARTINI and ATOMISTIC at the same time");

  // if MARTINI or unspecified, we need to read SCATLEN
  if(!atomistic) {
    SL_value.resize(size);
    ntarget=0;
    for(unsigned i=0; i<size; ++i) {
      if( !parseNumbered( "SCATLEN", i+1, SL_value[i]) ) break;
      ntarget++;
    }
    if( ntarget!=size ) error("found wrong number of SCATLEN");
  // if atomistic the values are tabulated
  } else if(atomistic) {
    SL_value.resize(size);
    tableASL(atoms, SL_value);
  }

  std::vector<double> expint;
  expint.resize( numq );
  ntarget=0;
  for(unsigned i=0; i<numq; ++i) {
    if( !parseNumbered( "EXPINT", i+1, expint[i] ) ) break;
    ntarget++;
  }
  bool exp=false;
  if(ntarget!=numq && ntarget!=0) error("found wrong number of EXPINT values");
  if(ntarget==numq) exp=true;
  if(getDoScore()&&!exp) error("with DOSCORE you need to set the EXPINT values");

  parse("SCALEINT", scale_int);

  if (exp) scale_int /= expint[0];

  // get the i=j term
  for (unsigned i=0; i<size; i++) {
    SL_rank += SL_value[i]*SL_value[i];
  }

  if(!getDoScore()) {
    for(unsigned i=0; i<numq; i++) {
      std::string num; Tools::convert(i,num);
      addComponentWithDerivatives("q-"+num);
      componentIsNotPeriodic("q-"+num);
    }
    if(exp) {
      for(unsigned i=0; i<numq; i++) {
        std::string num; Tools::convert(i,num);
        addComponent("exp-"+num);
        componentIsNotPeriodic("exp-"+num);
        Value* comp=getPntrToComponent("exp-"+num);
        comp->set(expint[i]);
      }
    }
  } else {
    for(unsigned i=0; i<numq; i++) {
      std::string num; Tools::convert(i,num);
      addComponent("q-"+num);
      componentIsNotPeriodic("q-"+num);
    }
    for(unsigned i=0; i<numq; i++) {
      std::string num; Tools::convert(i,num);
      addComponent("exp-"+num);
      componentIsNotPeriodic("exp-"+num);
      Value* comp=getPntrToComponent("exp-"+num);
      comp->set(expint[i]);
    }
  }

  // convert units to nm^-1
  for(unsigned i=0; i<numq; ++i) {
    q_list[i]=q_list[i]*10.0;    //factor 10 to convert from A^-1 to nm^-1
  }
  log<<"  Bibliography ";
  // TODO: change bibliography
  // if(martini) {
  //   log<<plumed.cite("Niebling, Björling, Westenhoff, J Appl Crystallogr 47, 1190–1198 (2014).");
  //   log<<plumed.cite("Paissoni, Jussupow, Camilloni, J Appl Crystallogr 52, 394-402 (2019).");
  // }
  // if(atomistic) {
  //   log<<plumed.cite("Fraser, MacRae, Suzuki, J. Appl. Crystallogr., 11, 693–694 (1978).");
  //   log<<plumed.cite("Brown, Fox, Maslen, O'Keefe, Willis, International Tables for Crystallography C, 554–595 (International Union of Crystallography, 2006).");
  // }
  log<< plumed.cite("Bonomi, Camilloni, Bioinformatics, 33, 3999 (2017)");
  log<<"\n";

  requestAtoms(atoms, false);
  if(getDoScore()) {
    setParameters(expint);
    Initialise(numq);
  }
  setDerivatives();
  checkRead();
}

void SANS::calculate_cpu(std::vector<Vector> &deriv)
{
  const unsigned size = getNumberOfAtoms();
  const unsigned numq = q_list.size();


  unsigned stride = comm.Get_size();
  unsigned rank   = comm.Get_rank();
  if(serial) {
    stride = 1;
    rank   = 0;
  }

  std::vector<double> sum(numq,0.);
  unsigned nt=OpenMP::getNumThreads();
  #pragma omp parallel num_threads(nt)
  {
    std::vector<Vector> omp_deriv(deriv.size());
    std::vector<double> omp_sum(numq,0.);

    #pragma omp for nowait
    for (unsigned i=rank; i<size-1; i+=stride) {
      Vector posi = getPosition(i);
      for (unsigned j=i+1; j<size ; j++) {
        Vector c_distances = delta(posi,getPosition(j));
        double m_distances = c_distances.modulo();
        c_distances = c_distances/m_distances/m_distances;
        for (unsigned k=0; k<numq; k++) {
          unsigned kdx=k*size;
          double qdist = q_list[k]*m_distances;
          double FSL = 2.*SL_value[i]*SL_value[j];
          double tsq = std::sin(qdist)/qdist;
          double tcq = std::cos(qdist);
          double tmp = FSL*(tcq-tsq);
          Vector dd  = c_distances*tmp;
          if(nt>1) {
            omp_deriv[kdx+i] -=dd;
            omp_deriv[kdx+j] +=dd;
            omp_sum[k]       += FSL*tsq;
          } else {
            deriv[kdx+i] -= dd;
            deriv[kdx+j] += dd;
            sum[k]       += FSL*tsq;
          }
        }
      }
    }
    #pragma omp critical
    if(nt>1) {
      for(unsigned i=0; i<deriv.size(); i++) deriv[i]+=omp_deriv[i];
      for(unsigned k=0; k<numq; k++) sum[k]+=omp_sum[k];
    }
  }

  if(!serial) {
    comm.Sum(&deriv[0][0], 3*deriv.size());
    comm.Sum(&sum[0], numq);
  }

  for (unsigned k=0; k<numq; k++) {
    sum[k] += SL_rank;
  }

  double normfactor = sum[0]*scale_int;
  for (unsigned i=0; i<deriv.size(); i++) deriv[i] = deriv[i]/normfactor;
  for (unsigned k=0; k<numq; k++) sum[k] /= normfactor;

  for (unsigned k=0; k<numq; k++) {
    std::string num; Tools::convert(k,num);
    Value* val=getPntrToComponent("q-"+num);
    val->set(sum[k]);
    if(getDoScore()) setCalcData(k, sum[k]);
  }
}

void SANS::calculate()
{
  if(pbc) makeWhole();

  const size_t size = getNumberOfAtoms();
  const size_t numq = q_list.size();

  std::vector<Vector> deriv(numq*size);
  calculate_cpu(deriv);

  if(getDoScore()) {
    /* Metainference */
    double score = getScore();
    setScore(score);
  }

  for (unsigned k=0; k<numq; k++) {
    const unsigned kdx=k*size;
    Tensor deriv_box;
    Value* val;
    if(!getDoScore()) {
      std::string num; Tools::convert(k,num);
      val=getPntrToComponent("q-"+num);
      for(unsigned i=0; i<size; i++) {
        setAtomsDerivatives(val, i, deriv[kdx+i]);
        deriv_box += Tensor(getPosition(i),deriv[kdx+i]);
      }
    } else {
      val=getPntrToComponent("score");
      for(unsigned i=0; i<size; i++) {
        setAtomsDerivatives(val, i, deriv[kdx+i]*getMetaDer(k));
        deriv_box += Tensor(getPosition(i),deriv[kdx+i]*getMetaDer(k));
      }
    }
    setBoxDerivatives(val, -deriv_box);
  }
}

void SANS::update() {
  // write status file
  if(getWstride()>0&& (getStep()%getWstride()==0 || getCPT()) ) writeStatus();
}

void SANS::tableASL(const std::vector<AtomNumber> &atoms, std::vector<double> &SL_value)
{
  std::map<std::string, unsigned> AA_map;
  AA_map["H"] = H;
  AA_map["D"] = D;
  AA_map["C"] = C;
  AA_map["N"] = N;
  AA_map["O"] = O;
  AA_map["P"] = P;
  AA_map["S"] = S;

  // assign scattering lengths based on Sears, Neutron News (1992)
  std::vector<double> atom_sl;
  atom_sl[H] = -3.7406;
  atom_sl[D] = 6.671;
  atom_sl[C] = 6.6511;
  atom_sl[N] = 9.37;
  atom_sl[O] = 5.803;
  atom_sl[P] = 5.13;
  atom_sl[S] = 2.804;

  auto* moldat=plumed.getActionSet().selectLatest<GenericMolInfo*>(this);

  if ( moldat ) {
    // cycle over the atoms to assign the scattering length based on atom type
    for (unsigned i=0; i<atoms.size(); ++i) {
      // get atom name
      std::string name = moldat->getAtomName(atoms[i]);
      char type;
      // get atom type
      char first = name.at(0);
      // GOLDEN RULE: type is first letter, if not a number
      if (!isdigit(first)) {
        type = first;
        // otherwise is the second
      } else {
        type = name.at(1);
      }
      std::string type_s = std::string(1,type);
      if(AA_map.find(type_s) != AA_map.end()) {
        const unsigned index=AA_map[type_s];
        SL_value[i] = atom_sl[index];
      } else {
        error("Wrong atom type "+type_s+" from atom name "+name+"\n");
      }
    }
  } else {
    error("MOLINFO DATA not found\n");
  }

}

}
}
