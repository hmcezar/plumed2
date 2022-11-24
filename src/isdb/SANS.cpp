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
 This class was originally written by Henrique Musseli Cezar,
 and is heavily based on the SAXS class. In fact it's basically
 the SAXS class that reads PARAMETERS with a single value instead
 of polynomial coeffs and with a different normalization. 
*/

#include "MetainferenceBase.h"
#include "core/ActionRegister.h"
#include "core/ActionSet.h"
#include "core/GenericMolInfo.h"
#include "tools/Communicator.h"
#include "tools/Pbc.h"

#include <map>
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif

// Splines algo https://stackoverflow.com/a/19216702/3254658
// good explanation on BC https://timodenk.com/blog/cubic-spline-interpolation/

namespace PLMD {
namespace isdb {

//+PLUMEDOC ISDB_COLVAR SANS
/*
Calculates SANS scattered intensity using the Debye equation.

Intensities are calculated for a set of scattering vectors set using QVALUE keywords that are numbered starting from 0.
Scattering lengths can be assigned for each atom/bead using the SCATLEN keywords.
The scattering lengths are automatically assigned to atoms using the ATOMISTIC flag reading the species from a PDB file.
For MARTINI models the user has to assign the scattering lengths using the SCATLEN keywords.
The output intensities are I(q)/I(q0), with q0 being the lowest scattering vector.
The intensities can be scaled using the SCALEINT keywords.
Experimental reference intensities can be added using the EXPINT keywords.
If the experimental intensities are provided, the output I(q) is automatically scaled so the calculated and experimental I(q0) match.
\ref METAINFERENCE can be activated using DOSCORE and the other relevant keywords.

\par Examples
in the following example the sans intensities for a martini model are calculated. scattering lengths
are obtained from the SCATLEN keywords.

\plumedfile
MOLINFO STRUCTURE=template.pdb

SANS ...
  LABEL=sans
  ATOMS=1-5

  MARTINI
  SCATLEN1=109.3554
  SCATLEN2=48.3351
  SCATLEN3=79.9724
  SCATLEN4=79.9724
  SCATLEN5=86.6434

  QVALUE1=0.011011 EXPINT1=0.56778
  QVALUE2=0.013054 EXPINT2=0.48948
  QVALUE3=0.016927 EXPINT3=0.5384
  QVALUE4=0.019104 EXPINT4=0.57098
  QVALUE5=0.020989 EXPINT5=0.63774
  QVALUE6=0.031085 EXPINT6=0.57009
  QVALUE7=0.040975 EXPINT7=0.54487
  QVALUE8=0.050954 EXPINT8=0.47315
  QVALUE9=0.060951 EXPINT9=0.39894
  QVALUE10=0.08093 EXPINT10=0.29655
  QVALUE11=0.09106 EXPINT11=0.26337
  QVALUE12=0.10079 EXPINT12=0.21249
  QVALUE13=0.11089 EXPINT13=0.16159
  QVALUE14=0.12037 EXPINT14=0.1333
  QVALUE15=0.13027 EXPINT15=0.09556

... SANS

PRINT ARG=(sans\.q-.*),(sans\.exp-.*) FILE=colvar STRIDE=1

\endplumedfile

*/
//+ENDPLUMEDOC

class SANS :
  public MetainferenceBase
{
private:
  enum { H, D, C, N, O, P, S, NTT };

  struct SplineCoeffs{
    double a;
    double b;
    double c;
    double d;
    double x;
  };

  bool                       pbc;
  bool                       serial;
  bool                       resolution;
  double                     SL_rank = 0.;
  double                     scale_int = 1.;
  int                        Nj = 10;
  std::vector<double>        q_list;
  std::vector<double>        SL_value;
  std::vector<double>        qj_list;
  std::vector<double>        sigma_res;

  void radial_distribution_hist();
  void calculate_cpu(std::vector<Vector> &deriv);
  void tableASL(const std::vector<AtomNumber> &atoms, std::vector<double> &SL_value);
  std::vector<SplineCoeffs> spline_coeffs(std::vector<double> &x, std::vector<double> &y);
  std::vector<double> interpolation(std::vector<SplineCoeffs> &coeffs, std::vector<double> &q_list, std::vector<double> &x);
  inline double spline(SplineCoeffs &coeffs, double &x);  
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
  keys.add("numbered","QVALUE","Selected scattering vector in Angstrom are given as QVALUE1, QVALUE2, ... .");
  keys.add("numbered","SCATLEN","Use SCATLEN keyword like SCATLEN1, SCATLEN2. These are the scattering lengths for the \\f$i\\f$th atom/bead.");
  keys.add("numbered","EXPINT","Add an experimental value for each q value.");
  keys.add("numbered","SIGMARES","Variance of Gaussian distribution describing the deviation in the scattering angle for each q value.");
  keys.add("compulsory","SCALEINT","1.0","SCALING value of the calculated data. Useful to simplify the comparison.");
  keys.add("compulsory","N","10","Number of points in the resolution function integral.");
  keys.addOutputComponent("q","default","the # SANS of q");
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

  // read experimental intensities
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

  // read sigma_res
  sigma_res.resize( numq );
  resolution=false;
  ntarget=0;
  for(unsigned i=0; i<numq; ++i) {
    if( !parseNumbered( "SIGMARES", i+1, sigma_res[i] ) ) break;
    ntarget++;
  }
  if(ntarget!=numq && ntarget!=0) error("found wrong number of SIGMARES values");
  if(ntarget==numq) resolution=true;

  parse("N", Nj);
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
  if(atomistic) {
    log<<plumed.cite("Sears, Neutron News, 3, 26 (1992)");
  }
  log<<plumed.cite("Bonomi, Camilloni, Bioinformatics, 33, 3999 (2017)");
  log<<"\n"; // TODO: add references for resolution function

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
            omp_deriv[kdx+i] -= dd;
            omp_deriv[kdx+j] += dd;
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
  
  // TODO: implement resolution correction
  if (resolution) {
    // get spline for scatering curve
    std::vector<SplineCoeffs> scatt_coeffs = spline_coeffs(q_list, sum);

    // artificially populate qj_list for debugging
    // qj_list.resize(100);
    // double dq = (q_list[q_list.size()-1] - q_list[0])/100;
    // for (unsigned i=0; i<100; i++) {
    //   qj_list[i] = q_list[0] + i*dq;
    // }
    std::vector<double> scatt_curve = interpolation(scatt_coeffs, q_list, qj_list);

    // for (unsigned i=0; i<qj_list.size(); i++) {
    //   std::cout << qj_list[i] << " " << scatt_curve[i] << std::endl;
    // }
    // get spline for the derivatives

  }

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

// compute resolution function
std::vector<Vector> resolution_function()
{
    int numq = q_list.size()
    std::vector<double> R(numq, Nj);

    for (unsigned i=0; i<numq; i++) {
      dq = 6*sigma_res[i]/Nj;
      for (unsigned j=0; j<Nj; j++) {
        std::vector<double> qj_list(Nj);
      }
    }
    // compute Bessel's function with Tools::bessel0(arg)
}

std::vector<double> SANS::interpolation(std::vector<SplineCoeffs> &coeffs, std::vector<double> &x)
{
  std::vector<double> curve(x.size());

  unsigned s = 0;
  for (unsigned i=0; i<x.size(); i++) {
    if ((x[i] >= q_list[s+1]) && (s+1 < q_list.size()-1)) s++;
    curve[i] = spline(coeffs[s], x[i]);
  }

  return curve;
}

inline double SANS::spline(SplineCoeffs &coeffs, double &x)
{
  double dx = x - coeffs.x;
  return coeffs.a + coeffs.b*dx + coeffs.c*dx*dx + coeffs.d*dx*dx*dx;
}

// natural bc cubic spline implementation from the Wikipedia algorithm
// modified from https://stackoverflow.com/a/19216702/3254658
std::vector<SANS::SplineCoeffs> SANS::spline_coeffs(std::vector<double> &x, std::vector<double> &y)
{
  unsigned n = x.size()-1;
  std::vector<double> a;
  a.insert(a.begin(), y.begin(), y.end());
  std::vector<double> b(n);
  std::vector<double> d(n);
  std::vector<double> h;

  for(unsigned i=0; i<n; i++)
    h.push_back(x[i+1]-x[i]);

  std::vector<double> alpha;
  alpha.push_back(0);
  for(unsigned i=1; i<n; i++)
    alpha.push_back( 3*(a[i+1]-a[i])/h[i] - 3*(a[i]-a[i-1])/h[i-1]  );

  std::vector<double> c(n+1);
  std::vector<double> l(n+1);
  std::vector<double> mu(n+1);
  std::vector<double> z(n+1);
  l[0] = 1;
  mu[0] = 0;
  z[0] = 0;

  for(unsigned i=1; i<n; i++) {
    l[i] = 2 *(x[i+1]-x[i-1])-h[i-1]*mu[i-1];
    mu[i] = h[i]/l[i];
    z[i] = (alpha[i]-h[i-1]*z[i-1])/l[i];
  }

  l[n] = 1;
  z[n] = 0;
  c[n] = 0;

  for(int j=n-1; j>=0; j--) {
    c[j] = z[j] - mu[j] * c[j+1];
    b[j] = (a[j+1]-a[j])/h[j]-h[j]*(c[j+1]+2*c[j])/3;
    d[j] = (c[j+1]-c[j])/3/h[j];
  }

  std::vector<SplineCoeffs> output_set(n);
  for(unsigned i=0; i<n; i++) {
    output_set[i].a = a[i];
    output_set[i].b = b[i];
    output_set[i].c = c[i];
    output_set[i].d = d[i];
    output_set[i].x = x[i];
  }

  return output_set;
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
