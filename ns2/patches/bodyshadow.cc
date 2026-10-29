#include <math.h>
#include <string.h>
#include "bodyshadow.h"
#include "mobilenode.h"
#include "wireless-phy.h"

static class BodyShadowClass : public TclClass {
 public:
  BodyShadowClass() : TclClass("Propagation/BodyShadow") {}
  TclObject* create(int, const char*const*) { return new BodyShadow; }
} class_bodyshadow;

BodyShadow::BodyShadow() : Shadowing() {
  bind("loss_db_", &loss_db_);
  bind("extra_std_db_", &extra_std_db_);
  bind("fresnel_scale_", &fresnel_scale_);
  body_radius_ = 0.2;
  rv_ = new RNG;
  rv_->set_seed(RNG::RAW_SEED_SOURCE, (int)seed_ + 1000);
}

int BodyShadow::command(int argc, const char*const* argv) {
  if (argc == 3 && strcmp(argv[1], "person") == 0) {
    // Register person node: $prop person $node
    MobileNode *node = (MobileNode*) TclObject::lookup(argv[2]);
    if (node) {
      people_.push_back(node);
      return TCL_OK;
    }
  }
  return Shadowing::command(argc, argv);
}

double BodyShadow::Pr(PacketStamp *t, PacketStamp *r, WirelessPhy *ifp) {
  // Base log-distance + Gaussian shadowing power from parent class
  double pr = Shadowing::Pr(t, r, ifp);

  MobileNode *tn = (MobileNode*) t->getNode();
  MobileNode *rn = (MobileNode*) r->getNode();
  if (!tn || !rn) {
    return pr;
  }

  tn->update_position();
  rn->update_position();

  double ax = tn->X(), ay = tn->Y();
  double bx = rn->X(), by = rn->Y();
  double L = sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay));
  if (L <= 1e-4) {
    return pr;
  }

  double lambda = 3e8 / ifp->getFreq();
  double extra_db = 0.0;
  double var_db2 = 0.0;

  for (size_t i = 0; i < people_.size(); i++) {
    MobileNode *p = people_[i];
    p->update_position();

    // Projection along link line (0..1)
    double s = ((p->X() - ax) * (bx - ax) + (p->Y() - ay) * (by - ay)) / (L * L);
    if (s <= 0.0 || s >= 1.0) {
      continue; // Beside or behind the transmitter / receiver
    }

    double dperp = fabs((p->X() - ax) * (by - ay) - (p->Y() - ay) * (bx - ax)) / L;
    double d1 = s * L;
    double d2 = (1.0 - s) * L;
    double rF = sqrt(lambda * d1 * d2 / (d1 + d2));
    double w = fresnel_scale_ * rF + body_radius_;
    double g = exp(-(dperp / w) * (dperp / w));

    extra_db += loss_db_ * g;
    var_db2 += (extra_std_db_ * extra_std_db_) * (g * g);
  }

  double noise = (var_db2 > 1e-6) ? rv_->normal(0.0, sqrt(var_db2)) : 0.0;
  double total_db_change = -extra_db + noise;

  return pr * pow(10.0, total_db_change / 10.0);
}
