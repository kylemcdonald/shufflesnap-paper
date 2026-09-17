// Experimental driver. Production core is copied without its Python bindings by build.py.
#include "build/core.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <numeric>
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}
int main(int argc,char** argv) {
 if(argc!=8) return 2;
 const std::string path=argv[1],mode=argv[3];
 const int side=std::stoi(argv[2]), threads=std::stoi(argv[4]), seed=std::stoi(argv[5]);
 const int passes=std::stoi(argv[6]); const bool shift=std::stoi(argv[7]);
 const int n=side*side;
 std::vector<double> pts(2*n);std::vector<int64_t> initial(n);
 std::ifstream input(path,std::ios::binary); input.read((char*)pts.data(),pts.size()*sizeof(double));input.read((char*)initial.data(),n*sizeof(int64_t));
 if(!input)throw std::runtime_error("input read failed");
 std::cout<<std::setprecision(17);
 const auto start=Clock::now();
 double grouping=0,solving=0; int random_passes=0;
 auto assignment=initial;
 if(mode!="baseline" && mode!="local" && mode!="alloffsets" && mode!="cycles") {
  std::vector<int> owner(n);for(int i=0;i<n;++i)owner[assignment[i]]=i;
  const auto xs=build_axis_coords(side);
  std::mt19937 rng(seed);
  std::vector<int> ks;
  int last=(side+5)/6;
  if(mode=="linear")for(int k=1;k<=last;++k)ks.push_back(k);
  else {for(int k=1;k<last;k*=2)ks.push_back(k);ks.push_back(last);}
  for(int k:ks) for(int rep=0;rep<passes;++rep) {
   auto t=Clock::now();
   std::vector<int> rb{0},cb{0};
   // Shift partition boundaries, clipping at both edges; never wrap the grid.
   int ro=shift && (passes==2 ? rep%2 : (rep/2)%2) ? side/(2*k):0;
   int co=shift && rep%2 ? side/(2*k):0;
   for(int i=0;i<=k;++i){int r=i*side/k+ro,c=i*side/k+co;if(r>0&&r<side)rb.push_back(r);if(c>0&&c<side)cb.push_back(c);}
   rb.push_back(side);cb.push_back(side);
   std::vector<int> cells;cells.reserve(n);std::vector<std::pair<int,int>> groups;
   groups.reserve(n/36+k*k);
   for(size_t r=1;r<rb.size();++r)for(size_t c=1;c<cb.size();++c){
    int begin=cells.size();
    for(int y=rb[r-1];y<rb[r];++y)for(int x=cb[c-1];x<cb[c];++x)cells.push_back(y*side+x);
    if(mode!="noshuffle") std::shuffle(cells.begin()+begin,cells.end(),rng);
    int end=cells.size();for(int j=begin;j<end;j+=36)groups.emplace_back(j,std::min(j+36,end));
   }
   grouping+=seconds(t);t=Clock::now();
   std::atomic<int> attempted{0},accepted{0};
   auto worker=[&](size_t begin,size_t end){
    int tries=0,wins=0;
    for(size_t g=begin;g<end;++g){auto [a,b]=groups[g];int count=b-a;if(count<2)continue; ++tries;
     std::array<int,36> ids{},solution{};std::array<double,1296> costs{};
     for(int i=0;i<count;++i)ids[i]=owner[cells[a+i]];
     for(int i=0;i<count;++i)for(int j=0;j<count;++j){int cell=cells[a+j];costs[i*count+j]=squared_distance(pts[2*ids[i]],pts[2*ids[i]+1],xs[cell%side],xs[cell/side]);}
     if(!solve_square_jv_small(count,costs.data(),solution.data()))std::terminate();
     double before=0,after=0;for(int i=0;i<count;++i){before+=costs[i*count+i];after+=costs[i*count+solution[i]];}
     if(before-after<=1e-12*before)continue;
     ++wins;
     for(int i=0;i<count;++i){int cell=cells[a+solution[i]];assignment[ids[i]]=cell;owner[cell]=ids[i];}
    }
    attempted.fetch_add(tries);accepted.fetch_add(wins);
   };
   int nt=resolve_thread_count(threads,groups.size());std::vector<std::thread> workers;
   if(nt==1)worker(0,groups.size());else {for(int i=0;i<nt;++i)workers.emplace_back(worker,groups.size()*i/nt,groups.size()*(i+1)/nt);for(auto& w:workers)w.join();}
   solving+=seconds(t);++random_passes;
   double cost=0;for(int i=0;i<n;++i){int cell=assignment[i];cost+=squared_distance(pts[2*i],pts[2*i+1],xs[cell%side],xs[cell/side]);}
   std::cout<<"trace,"<<seconds(start)<<","<<cost<<",random,"<<k<<","<<attempted<<","<<accepted<<"\n";
  }
 }
 double prep=seconds(start);
 std::vector<int64_t> schedule;
 if(mode=="baseline" || mode=="alloffsets" || mode=="cycles"){int s=1;while(6*s<side)s*=2;for(;s>=1;s/=2){schedule.push_back(s);schedule.push_back(s);}}
 else schedule={1,1};
 if(mode=="cycles"){auto again=schedule;schedule.insert(schedule.end(),again.begin(),again.end());}
 auto result=run_cleanup(pts.data(),n,assignment.data(),side,side,(mode=="local" || mode=="noshuffle") ? 30 : 120,6,0,threads,schedule,true,mode=="alloffsets",{});
 for(size_t i=0;i<result.round_costs.size();++i)std::cout<<"trace,"<<prep+result.round_elapsed_s[i]<<","<<result.round_costs[i]<<",polish,"<<result.round_strides_row[i]<<"\n";
 std::cout<<"summary,"<<seconds(start)<<","<<result.final_cost<<","<<grouping<<","<<solving<<","<<result.elapsed_s<<","<<random_passes<<","<<result.rounds_completed<<","<<result.converged<<"\n";
 std::ofstream output(path+".assignment",std::ios::binary);output.write((char*)result.assignment.data(),n*sizeof(int64_t));
}
