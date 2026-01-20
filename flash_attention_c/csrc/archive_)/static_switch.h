#pragma once

// #define FP_SWITCH(TYPE, NAME, ...)                                             \
//   [&] {                                                                        \
//     const auto &the_type = TYPE;                                               \
//     constexpr const char *at_dispatch_name = NAME;                             \
//     at::ScalarType _st = ::detail::scalar_type(the_type);                      \
//     switch (_st) {                                                             \
//     case at::ScalarType::Double: {                                             \
//       using scalar_t =                                                         \
//           c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Double>;             \
//       return __VA_ARGS__();                                                    \
//     }                                                                          \
//     case at::ScalarType::Float: {                                              \
//       using scalar_t = c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Float>; \
//       return __VA_ARGS__();                                                    \
//     }                                                                          \
//     case at::ScalarType::Half: {                                               \
//       using scalar_t = c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Half>;  \
//       return __VA_ARGS__();                                                    \
//     }                                                                          \
//     case at::ScalarType::BFloat16: {                                           \
//       using scalar_t =                                                         \
//           c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::BFloat16>;           \
//       return __VA_ARGS__();                                                    \
//     }                                                                          \
//     default: {                                                                 \
//       TORCH_CHECK(false, '"', at_dispatch_name, "\" not implemented for '",    \
//                   toString(_st), "'");                                         \
//     }                                                                          \
//     }                                                                          \
//   }()

// TODO: only half and bf16 are supported
#define FP_SWITCH(TYPE, NAME, ...)                                             \
  [&] {                                                                        \
    const auto &the_type = TYPE;                                               \
    constexpr const char *at_dispatch_name = NAME;                             \
    at::ScalarType _st = ::detail::scalar_type(the_type);                      \
    switch (_st) {                                                             \
    case at::ScalarType::Float: {                                              \
      using scalar_t = c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Float>; \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case at::ScalarType::Half: {                                               \
      using scalar_t =                                                         \
          c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Half>;               \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case at::ScalarType::BFloat16: {                                           \
      using scalar_t =                                                         \
          c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::BFloat16>;           \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    default: {                                                                 \
      TORCH_CHECK(false, '"', at_dispatch_name, "\" not implemented for '",    \
                  toString(_st), "'");                                         \
    }                                                                          \
    }                                                                          \
  }()

