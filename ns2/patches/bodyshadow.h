#ifndef ns_bodyshadow_h
#define ns_bodyshadow_h

#include <vector>
#include "shadowing.h"

class MobileNode;

class BodyShadow : public Shadowing {
 public:
  BodyShadow();
  virtual double Pr(PacketStamp *tx, PacketStamp *rx, WirelessPhy *ifp);
  virtual int command(int argc, const char*const* argv);

 protected:
  std::vector<MobileNode*> people_;
  double loss_db_;
  double extra_std_db_;
  double fresnel_scale_;
  double body_radius_;
  RNG *rv_;
};

#endif // ns_bodyshadow_h
