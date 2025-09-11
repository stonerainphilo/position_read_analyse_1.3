// based in par on main41.cc is a part of the PYTHIA event generator.
// Copyright (C) 2015 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Author: Simon Knapen
// This program simulates B->K X, with X a real scalar decaying to muons.
// Output is in reduced hepmc format



// ----------------------------2025 3 29---------------------------------------

// 添加了对于pp对撞，产生LLP粒子的计数

// -------------------------------------------------------------------

#include "Pythia8/Pythia.h"
#include "Pythia8/ParticleData.h"
#include "Pythia8Plugins/HepMC2.h"
#include "dataframe.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <typeinfo>
#include <vector>

using namespace Pythia8;
using namespace std;
// const double c_mm = 3.0e5;

double calculate_Gamma_aka_minwidth(double ctau_In_Meters) {
    const double c = 3.0e8;
    const double hbar = 6.58212e-25;
    double gamma_aka_minwidth = hbar / (ctau_In_Meters / c);
    return gamma_aka_minwidth;
}

std::string doubleToScientificString(double value, int precision = 5) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(precision) << value;
    return oss.str();
}

void printDoubleInScientific(double value, int precision = 5) {
    std::cout << std::scientific << std::setprecision(precision) << value << std::endl;
}


Event TrimEvent(Event event, Event trimmedevent)
{
    int              trimmedAprimepos = 0;
    int              trimmedmotherpos = 0;
    int              trimmedsisterpos = 0;
    std::vector<int> motherpos;
    std::vector<int> sisterpos;
    int              sisterposlast;

    for (int i = 0; i < event.size(); ++i) {
        // cout << event[i].id() << endl;
        if (event[i].id() == 999999)
        {
            // event.id is wrong
            // Find mother (the B meson)
            motherpos = event[i].motherList();
            trimmedevent.append(event[motherpos[0]]);
            trimmedmotherpos = trimmedevent.size() - 1;   // record mother pos in new eventhiyuzhe_Ubuntu
            trimmedevent[trimmedmotherpos].mothers(0, 0); // fix the mother of the B to be the header

            // Find sister (the K meson)
            sisterpos     = event[i].sisterList();
            sisterposlast = event[sisterpos[0]].iBotCopyId();
            if (event[sisterposlast].isFinal()) {
                trimmedevent.append(event[sisterposlast]);
            }
            else { // If K0, go one more level down, to K_L or K_S
                sisterposlast = (event[sisterposlast].daughterList())[0];
                trimmedevent.append(event[sisterposlast]);
            }
            trimmedsisterpos = trimmedevent.size() - 1;
            trimmedevent[trimmedsisterpos].mothers(trimmedmotherpos, 0);

            // Now add the A' to the event
            trimmedevent.append(event[i]);
            trimmedAprimepos = trimmedevent.size() - 1;
            trimmedevent[trimmedAprimepos].mothers(trimmedmotherpos, 0);

            // Search for the daughters of the A'
            for (int j = 0; j < event.size(); ++j) {
                if (event[j].isAncestor(i) && event[j].isFinal()) {
                    event[j].mothers(trimmedAprimepos, 0);
                    trimmedevent.append(event[j]);
                }
            }
        }
    }
    return trimmedevent;
}

// assuming input parameters: (mass, ctau, br, seed)
int main(int argc, char *argv[])
{
    if (argc != 8)
    {
        std::cerr << "assuming input parameters: "
                <<"(mass, ctau, br, seed, Output_dir tanbeta Decay_Width"
                <<")";
        return -1;
    }
    std::cout << std::scientific <<std::setprecision(10);
    double    mA   = atof(argv[1]);
    double    ctau = atof(argv[2]);
    string ctau_string = std::to_string(ctau);
    double br = atof(argv[3]);
    long long seed = atoll(argv[4]);
    string dir_path = argv[5];

    double tanb = atof(argv[6]);
    double mWidth = atof(argv[7]);
    // std::cout << doubleToScientificString(br) <<std::endl;
    std::string outfilename = dir_path + "mass_" + doubleToScientificString(mA, 2)
                            + "_ctau_" + doubleToScientificString(ctau, 2)
                            + "_br_" + doubleToScientificString(br, 2)
                            + "_seed_" + std::to_string(seed)
                            + ".csv";
    // std::cout << outfilename << endl;
    // WARN: redirecting stdout so we dont see bullshit startup banner
    // WARN: /dev/null only works on posix systems.
    // std::freopen("/Users/shiyuzhe/Documents/University/LLP/Second_Term/pythia8/examples/main41_error_log.txt", "w", stdout);
    DataFrame df;
    Pythia pythia;
    Event &event = pythia.event;
    ParticleData pdata;
    // ParticleData pdata521;
    // ParticleData pdata511;
    // ParticleData pdata999999;
    ParticleDataEntry pdataE;
    // ParticleDataEntry pdataE521;
    // ParticleDataEntry pdataE511;
    // ParticleDataEntry pdataE999;
    DecayChannel DC;
    // DecayChannel DC521; 
    // DecayChannel DC511;
    // DecayChannel DC999999;
    pythia.readString("Print:quiet = on");
    pythia.readString("Random:setSeed = on");
    pythia.readString("Beams:frameType = 2"); // Set the Still Target
    pythia.readString("Random:seed = " + std::to_string(seed));
    // pythia.readString("Beams:eCM = 14000.");//  total Energy in Gev
    pythia.readString("Beams:idA = 2212"); // proton
    pythia.readString("Beams:idB = 2212"); // proton
    pythia.readString("Beams:eA = 400"); // beam A
    pythia.readString("Beams:eB = 0"); // beam B
    pythia.readString("HardQCD:hardbbbar  = on");
    // pythia.readString("HardQCD:all = on");
    pythia.readString("999999:all = GeneralResonance void 0 0 0 " + doubleToScientificString(mA) + " " + doubleToScientificString(mWidth) + " 0 20 " + doubleToScientificString(ctau));
    // pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_Hmumu, 5) + " 101 13 -13"); // to mu-mu+
    // pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_Hee, 5) + " 101 11 -11"); // to e-e+
    // pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_HKK, 5) + " 101 311 311"); // to Kaon^0 Kaon^0
    // pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_HPiPi, 5) + " 101 111 111"); // to pion pion
    // pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_Htautau, 5) + " 101 15 -15"); // to tau^- tau^+
    // pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_HGluon, 5) + " 101 21 21"); // gg
    // pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_Hgaga, 5) + " 101 22 22"); //gamma gamma
    // pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_H4Pi, 5) + " 101 111 111 111 111"); // to 4 pion (uncharged)
    // pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_Hss, 5) + " 101 3 -3"); // to s sbar
    // pythia.readString("999999:addchannel = 1 " + doubleToScientificString(Br_Hcc, 5) + " 101 4 -4"); // to c cbar
    pythia.readString("999999:isResonance = true");
    // pythia.readString("ResonanceWidths:minWidth = 1e-30" + doubleToScientificString(calculate_Gamma_aka_minwidth(ctau)));
    // pythia.readString("ResonanceWidths:minWidth = 1e-30");
    pythia.readString("999999:tauCalc = false");
    
    pythia.readString("999999:onMode = 1");
    pythia.readString("999999:MayDecay = true");
    // pythia.readString("ResonanceWidths:enhance = 999999 " + doubleToScientificString(br/2));
    // cout << "ok after prepare_pick " << endl;
    pythia.readString("521:oneChannel = 1 1 101 999999 321");
    pythia.readString("511:oneChannel = 1 1 101 999999 311");
    pythia.init();
    // cout << pythia.info.weight() << endl;
    // boostVec = 
    // pythia.info.
    Event trimmedevent;
    // HepMC::GenEvent *hepmcevt = new HepMC::GenEvent(HepMC::Units::GEV, HepMC::Units::CM);
    // WARN: redirecting stdout output to the output file so we can get the data.
    std::cout<<"ok before csv"<<endl;
    // std::freopen(outfilename.c_str(), "w", stdout);
    // int countDecay521To999999 = 0;
    // int count521Decays = 0;
    // int count511Decays = 0;
    // int countDecay511To999999 = 0;
    // int countDecays = 0;
    // int countDecaysTo999999 = 0; 
    // size_t countDecay521To999999 = 0;
    // size_t count521Decays = 0;
    // size_t count511Decays = 0;
    // size_t countDecay511To999999 = 0;
    long int number_LLP = 0;
    long int number_of_production = 0;
    for (int iEvent = 0; iEvent < 100000; ++iEvent)
    {
        if (!pythia.next())
            continue;
        
        trimmedevent = event;
        trimmedevent.reset();
        trimmedevent = TrimEvent(event, trimmedevent);
        // double BetaZ = 0.9977;
        // trimmedevent.bst(0, 0, -BetaZ);
        // trimmedevent.list(true, true, 5);
        if (trimmedevent.size() == 0)
        {
            std::cout << "Trimmed event is empty for event " << iEvent << std::endl;
            continue;
        }
        // df.addRow({{id, ev.id()}})



        bool hasLLPInEvent = false;

        for (const auto &ev : trimmedevent) {

            if (ev.id() == 999999){
                number_LLP++;
                {
                if (!hasLLPInEvent)
                {
                    number_of_production++; // 如果是当前事件中第一次发现 id == 999999 的粒子
                    hasLLPInEvent = true;  // 标记为 true，避免重复计数
                }
                Vec4 boostedVDec = ev.vDec();
                boostedVDec.bst(0, 0, -0.9977);
                // cout << "There is a LLP" << endl;
                // cout << doubleToScientificString(ev.tau()) << endl;
                df.addRow({
                    {"id", ev.id()},
                    // {"p_x", ev.px()},
                    // {"p_y", ev.py()},
                    // {"p_z", ev.pz()},
                    // {"e", ev.e()},
                    {"m", ev.m()},
                    // {"xProd", ev.xProd()},
                    // {"yProd", ev.yProd()},
                    // {"zProd", ev.zProd()},
                    // {"tProd", ev.tProd()},
                    {"tau", ev.tau()},
                    //    {"br", br},
                    {"seed", seed},
                    {"decay_pos_x", boostedVDec.px()},
                    {"decay_pos_y", boostedVDec.py()},
                    {"decay_pos_z", boostedVDec.pz()}, // Now in lab frame
                    {"decay_pos_t", boostedVDec.pT()},
                    // {"decay_pos_x", ev.xDec()},
                    // {"decay_pos_y", ev.yDec()},
                    // {"decay_pos_z", ev.zDec()},
                    // {"decay_pos_t", ev.tDec()},
                    {"mass_input", mA},
                    {"tau_input", ctau},
                    {"theta", tanb},
                    {"LLP_number_per_ev", number_LLP},
                    {"total_events", iEvent},
                    {"number_of_production", number_of_production},
                    {"Cross_section_fb", pythia.info.sigmaGen() * 1e12},
                    // {"sigma_without_unit", number_LLP/ iEvent}
                        //    {"br",
                        //     (countDecay511To999999 + countDecay521To999999)/(count511Decays + count521Decays)
                        //     }
                    });
            }

            //         printf(
            //             "id=%i, "
            //             "px=%.8f, py=%.8f, pz=%.8f, "
            //             "e=%.8f, m=%.8f, "
            //             "xProd=%.8f, yProd=%.8f, zProd=%.8f, "
            //             "tProd=%.8f, tau=%.8f\n",
            //             ev.id(),
            //             ev.px(), ev.py(), ev.pz(),
            //             ev.e(), ev.m(),
            //             ev.xProd(), ev.yProd(), ev.zProd(),
            //             ev.tProd(), ev.tau());
        }
        // event.list(true, true);
        // trimmedevent.list(true, true, 5);
    }}
    df.toCSV(outfilename);

    // auto file = std::ofstream(outfilename);

    // WARN: explicitly closing stdout (now a file)
    fclose(stdout);
    return 0;
}