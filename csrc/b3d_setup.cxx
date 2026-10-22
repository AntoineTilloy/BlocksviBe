// b3d_setup: blocks_3d's own setup stages for the GPU chain (blocksvibe), written to one binary file:
//   * the M matrices (once per four-point parity: shared by all four-point structures q4 of a
//     (j12, j43, Delta_12, Delta_43) group), and
//   * h_infinity (the recursion's initial data) for every requested (q4, sign) and lambda order n,
// computed by blocks_3d's own functions (linked from its source tree) at blocks_3d's working precision,
// so the GPU chain starts from bit-identical inputs.  The rest of blocks_3d (radial recursion, residues
// -> derivatives, pole shifting, write_output) is not called.
//
// Usage (internal; called by blocksvibe/setup.py):
//   b3d_setup out=<file> precision=<bits> threads=<n> two_js=a,b,c,d two_j12=<k> two_j43=<k>
//             delta_12=<str> delta_43=<str> order=<k> lambda=<k> two_j_max=<k> radial=<0|1>
//             q4=a,b,c,d,sign [q4=... ...]           (all spins doubled, i.e. two_j)
//
// Binary layout (little endian int64 words unless noted), K = limbs per number:
//   "B3DSETUP" K
//   nM; nM x [ptype pvalue delta(0=12,1=43) two_jNN prime_pseudo_parity two_j rows cols]; then all
//       matrix entries row-major, as numbers
//   ntasks; per task: q4[4] sign pp00 pp01 pp10 pp11 nn; per n: n count; count x [pp m j a b]; count numbers
// number = [size exp limb_0 .. limb_{K-1}]: GMP mpf with the top K limbs (limb_{K-1} most significant,
//   zero-padded below), value = sign(size) * sum_t limb_t 2^(64 t) * 2^(64 (exp - K)).
#include "lambda_log_r_tables/Cache.hxx"
#include "compute_pseudo_parity_types.hxx"
#include "timers/Timers.hxx"

#include <cstdio>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>

void fill_binomials(const Bigfloat &delta_12, const Bigfloat &delta_43,
                    const std::array<int64_t, 4> &q4, const int64_t &order,
                    std::vector<Bigfloat> &binomial_a,
                    std::vector<Bigfloat> &binomial_ab,
                    std::vector<Bigfloat> &binomial_b,
                    std::vector<Bigfloat> &binomial_bb);

std::vector<std::vector<Bigfloat>>
h_infinity_double_series(const int64_t &order,
                         const std::vector<Bigfloat> &binomial_a,
                         const std::vector<Bigfloat> &binomial_ab,
                         const std::vector<Bigfloat> &binomial_b,
                         const std::vector<Bigfloat> &binomial_bb);

std::vector<std::vector<std::vector<std::vector<std::vector<Bigfloat>>>>>
h_infinity(const std::array<int64_t, 4> &two_js,
           const std::vector<std::vector<Bigfloat>> &series,
           const int64_t &two_j12, const int64_t &two_j43,
           const std::array<int64_t, 4> &q4, const int64_t &i,
           const int64_t &two_j_max, const int64_t &order, const int64_t &n,
           const F_SO3_Basis_Derivative &f_so3_basis_derivative,
           const size_t &num_threads, Timers &timers);

namespace
{
  std::array<int64_t, 4> ints4(const std::string &s, int64_t *extra = nullptr)
  {
    std::array<int64_t, 4> r{};
    std::stringstream ss(s);
    std::string item;
    for(int i = 0; i < 4; ++i)
      {
        std::getline(ss, item, ',');
        r[i] = std::stoll(item);
      }
    if(extra)
      {
        std::getline(ss, item, ',');
        *extra = std::stoll(item);
      }
    return r;
  }

  struct Writer
  {
    std::ofstream os;
    int64_t K;
    void word(int64_t w) { os.write(reinterpret_cast<const char *>(&w), 8); }
    void number(const Bigfloat &x)
    {
      const __mpf_struct *v = x.backend().data();
      const int64_t size = v->_mp_size, n = std::abs(size);
      word(size);
      word(n == 0 ? 0 : v->_mp_exp);
      for(int64_t t = 0; t < K; ++t)
        {
          const int64_t src = n - K + t;   // top K limbs, zero-padded below
          const uint64_t limb = (src >= 0 && src < n) ? v->_mp_d[src] : 0;
          os.write(reinterpret_cast<const char *>(&limb), 8);
        }
    }
  };
}

int main(int argc, char *argv[])
{
  try
    {
      std::map<std::string, std::string> kv;
      std::vector<std::string> q4s;
      for(int a = 1; a < argc; ++a)
        {
          std::string s(argv[a]);
          auto eq = s.find('=');
          if(eq == std::string::npos)
            throw std::runtime_error("bad argument: " + s);
          if(s.substr(0, eq) == "q4")
            q4s.push_back(s.substr(eq + 1));
          else
            kv[s.substr(0, eq)] = s.substr(eq + 1);
        }
      const size_t precision_base_2 = std::stoul(kv.at("precision")),
                   num_threads = std::stoul(kv.at("threads"));
      // as blocks_3d's handle_input: round up to whole limbs, set the decimal default precision
      const size_t rounded((((precision_base_2 - 1) / (sizeof(mp_limb_t) * 8)) + 1)
                           * sizeof(mp_limb_t) * 8);
      boost::multiprecision::mpf_float::default_precision(
        boost::multiprecision::detail::digits2_2_10(rounded));

      const std::array<int64_t, 4> two_js(ints4(kv.at("two_js")));
      const int64_t two_j12 = std::stoll(kv.at("two_j12")), two_j43 = std::stoll(kv.at("two_j43")),
                    order = std::stoll(kv.at("order")), lambda = std::stoll(kv.at("lambda")),
                    two_j_max = std::stoll(kv.at("two_j_max"));
      const bool radial = kv.at("radial") == "1";
      const Bigfloat delta_12(kv.at("delta_12")), delta_43(kv.at("delta_43"));

      Writer w;
      w.os.open(kv.at("out"), std::ios::binary);
      w.K = Bigfloat(1).backend().data()->_mp_prec + 1;
      w.os.write("B3DSETUP", 8);
      w.word(w.K);

      Timers timers(false);
      Factorial factorial;
      factorial.initialize(two_j_max + 2 * order + 2);
      Clebsch_Gordan clebsch_gordan_cache;
      clebsch_gordan_cache.initialize(two_j12, two_j_max + 2 * order, factorial);
      clebsch_gordan_cache.initialize(two_j43, two_j_max + 2 * order, factorial);
      Pole_Types pole_types;
      pole_types.initialize(two_j_max, two_j12, two_j43, order);

      std::vector<std::pair<Q4pm, int64_t>> tasks;      // (q4pm, four-point parity)
      std::set<int64_t> parities;
      for(auto &s : q4s)
        {
          int64_t sign;
          auto v = ints4(s, &sign);
          tasks.emplace_back(Q4pm(v, sign), four_point_parity(two_js, v));
          parities.insert(tasks.back().second);
        }

      // M matrices: depend on q4 only through the four-point parity; one cache for all parities
      M_Matrix m_matrix(delta_12, delta_43);
      for(auto parity : parities)
        m_matrix.initialize(two_j12, two_j43, two_j_max, parity, order, pole_types,
                            clebsch_gordan_cache, factorial);
      w.word(m_matrix.cache.size());
      for(auto &[key, mat] : m_matrix.cache)
        {
          auto &[pole, d, two_jNN, ppp, two_j] = key;
          for(int64_t x : {int64_t(pole.type), pole.value, int64_t(d), two_jNN, ppp, two_j,
                           int64_t(mat.size()), int64_t(mat.empty() ? 0 : mat[0].size())})
            w.word(x);
        }
      for(auto &[key, mat] : m_matrix.cache)
        for(auto &row : mat)
          for(auto &x : row)
            w.number(x);

      w.word(tasks.size());
      for(auto &[q4pm, parity] : tasks)
        {
          std::vector<int64_t> lambda_orders;
          for(int64_t n = (1 - q4pm.i) / 2; n <= (radial ? (1 - q4pm.i) / 2 : lambda); n += 2)
            lambda_orders.push_back(n);
          // q4-dependent caches, as in lambda_log_r_tables
          Wigner_Cache wigner_cache;
          F_Q_Basis_Derivative f_q_basis_derivative;
          F_W_Conversion_Matrix f_w_conversion_matrix;
          F_SO3_Basis_Derivative f_SO3_basis_derivative;
          wigner_cache.initialize(two_j_max + 2 * order, two_js, q4pm.vec, lambda, factorial);
          f_q_basis_derivative.initialize(two_js, two_j_max + 2 * order, lambda, wigner_cache, timers);
          f_SO3_basis_derivative.initialize(two_js, two_j_max + 2 * order, two_j12, two_j43, q4pm.vec,
                                            lambda, clebsch_gordan_cache, factorial,
                                            f_q_basis_derivative, f_w_conversion_matrix);
          const auto pp(compute_pseudo_parity_types(two_js, q4pm.vec, {two_js[0], two_js[1], two_j12},
                                                    {two_js[3], two_js[2], two_j43}));
          for(int64_t x : q4pm.vec)
            w.word(x);
          w.word(q4pm.i);
          for(int64_t x : {pp[0][0], pp[0][1], pp[1][0], pp[1][1], int64_t(lambda_orders.size())})
            w.word(x);

          // h_infinity, as in conformal_block_derivatives
          std::vector<Bigfloat> binomial_a, binomial_ab, binomial_b, binomial_bb;
          fill_binomials(delta_12, delta_43, q4pm.vec, order, binomial_a, binomial_ab, binomial_b,
                         binomial_bb);
          const auto series(h_infinity_double_series(order, binomial_a, binomial_ab, binomial_b,
                                                     binomial_bb)),
            series_b(h_infinity_double_series(order, binomial_ab, binomial_a, binomial_bb, binomial_b));
          std::array<int64_t, 4> neg_q4;
          for(int i = 0; i < 4; ++i)
            neg_q4[i] = -q4pm.vec[i];
          const int64_t factor(q4pm.i * (std::accumulate(two_js.begin(), two_js.end(), 0) % 4 == 0 ? 1 : -1));
          for(int64_t n : lambda_orders)
            {
              auto h_inf(h_infinity(two_js, series, two_j12, two_j43, q4pm.vec, 0, two_j_max, order, n,
                                    f_SO3_basis_derivative, num_threads, timers));
              const auto h_temp(h_infinity(two_js, series_b, two_j12, two_j43, neg_q4, 1, two_j_max, order,
                                           n, f_SO3_basis_derivative, num_threads, timers));
              std::vector<std::array<int64_t, 5>> idx;
              for(size_t p(0); p != h_inf.size(); ++p)
                for(size_t m(0); m != h_inf[p].size(); ++m)
                  for(size_t j(0); j != h_inf[p][m].size(); ++j)
                    for(size_t a(0); a != h_inf[p][m][j].size(); ++a)
                      for(size_t b(0); b != h_inf[p][m][j][a].size(); ++b)
                        {
                          auto &h(h_inf[p][m][j][a][b]);
                          h += factor * h_temp[p][m][j][a][b];
                          h *= 0.5;
                          idx.push_back({int64_t(p), int64_t(m), int64_t(j), int64_t(a), int64_t(b)});
                        }
              w.word(n);
              w.word(idx.size());
              for(auto &t : idx)
                for(auto x : t)
                  w.word(x);
              for(auto &t : idx)
                w.number(h_inf[t[0]][t[1]][t[2]][t[3]][t[4]]);
            }
        }
      w.os.close();
      if(!w.os)
        throw std::runtime_error("write failed: " + kv.at("out"));
    }
  catch(std::exception &e)
    {
      std::cerr << "b3d_setup error: " << e.what() << "\n";
      return 1;
    }
  return 0;
}
