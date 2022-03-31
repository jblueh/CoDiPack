#pragma once

#include <complex>

#include "../binaryExpression.hpp"
#include "../expressionInterface.hpp"
#include "../expressionMemberOperations.hpp"
#include "../real/unaryOperators.hpp"
#include "../unaryExpression.hpp"
#include "../../misc/macros.hpp"
#include "../../config.h"
#include "../activeType.hpp"
#include "../aggregatedExpressionType.hpp"
#include "allOperators.hpp"
#include "../real/binaryOperators.hpp"
#include "../../misc/self.hpp"
#include "../../traits/computationTraits.hpp"

/** \copydoc codi::Namespace */
namespace codi {

  template<typename T_Real, typename T_Impl>
  struct ExpressionMemberOperations<T_Real, T_Impl, typename std::enable_if<std::is_same<T_Real, std::complex<typename T_Real::value_type>>::value>::type> {

      using Real = CODI_DD(T_Real, std::complex<double>);
      using Impl = CODI_DD(T_Impl, CODI_T(ExpressionInterface<Real, void>));

      using InnerType = typename Real::value_type;

      using ExpressionComplexReal = UnaryExpression<InnerType, Impl, OperationComplexReal>;

      ExpressionComplexReal real() const {
        return ExpressionComplexReal(cast());
      }

      using ExpressionComplexImag = UnaryExpression<InnerType, Impl, OperationComplexImag>;

      ExpressionComplexImag imag() const {
        return ExpressionComplexImag(cast());
      }

    private:
      CODI_INLINE Impl const& cast() const {
        return static_cast<Impl const&>(*this);
      }
  };

  namespace RealTraits {
    template<typename T_InnerReal>
    struct AggregatedTypeTraits<std::complex<T_InnerReal>>
        : public ArrayAggregatedTypeTraitsBase<std::complex<T_InnerReal>, T_InnerReal, std::complex<RealTraits::Real<T_InnerReal>>, 2> {};
  }


  template<typename T_InnerActive, typename T_Impl = Self>
  struct ActiveComplex
      : public AggregatedExpressionType<std::complex<typename T_InnerActive::Real>, T_InnerActive,
                                        ReturnSelf<T_Impl, ActiveComplex<T_InnerActive, T_Impl>>> {

    public:

      using InnerActive = CODI_DD(T_InnerActive, CODI_T(ActiveType<CODI_ANY>));
      using Impl = CODI_DD(CODI_T(ReturnSelf<T_Impl, ActiveComplex>), ActiveComplex);

      using InnerReal = typename InnerActive::Real;
      using Real = std::complex<InnerReal>;
      using PassiverInnerReal = RealTraits::PassiveReal<InnerReal>;

      using Base = AggregatedExpressionType<Real, InnerActive, Impl>;

      using value_type = InnerActive;

      using Base::Base;
      using Base::operator =;

      template<typename ArgR>
      ActiveComplex(
          ExpressionInterface<InnerReal, ArgR> const& argR) :
        Base()
      {
        Base::template array<0>() = argR;
      }

      ActiveComplex(
          PassiverInnerReal const& argR) :
        Base()
      {
        Base::template array<0>() = argR;
      }

      template<typename ArgR, typename ArgI>
      ActiveComplex(
          ExpressionInterface<InnerReal, ArgR> const& argR,
          ExpressionInterface<InnerReal, ArgI> const& argI) :
        Base()
      {
        Base::template array<0>() = argR;
        Base::template array<1>() = argI;
      }

      template<typename ArgI>
      ActiveComplex(
          PassiverInnerReal const& argR,
          ExpressionInterface<InnerReal, ArgI> const& argI) :
        Base()
      {
        Base::template array<0>() = argR;
        Base::template array<1>() = argI;
      }

      template<typename ArgR>
      ActiveComplex(
          ExpressionInterface<InnerReal, ArgR> const& argR,
          PassiverInnerReal const& argI) :
        Base()
      {
        Base::template array<0>() = argR;
        Base::template array<1>() = argI;
      }

      ActiveComplex(
          PassiverInnerReal const& argR,
          PassiverInnerReal const& argI) :
        Base()
      {
        Base::template array<0>() = argR;
        Base::template array<1>() = argI;
      }

      /// Operator += for expressions.
      template<typename Rhs>
      CODI_INLINE Impl& operator+=(ExpressionInterface<Real, Rhs> const& rhs) {
        return Base::cast() = (Base::cast() + rhs);
      }

      /// Operator -= for expressions.
      template<typename Rhs>
      CODI_INLINE Impl& operator-=(ExpressionInterface<Real, Rhs> const& rhs) {
        return Base::cast() = (Base::cast() - rhs);
      }

      /// Operator *= for expressions.
      template<typename Rhs>
      CODI_INLINE Impl& operator*=(ExpressionInterface<Real, Rhs> const& rhs) {
        return Base::cast() = (Base::cast() * rhs);
      }

      /// Operator /= for expressions.
      template<typename Rhs>
      CODI_INLINE Impl& operator/=(ExpressionInterface<Real, Rhs> const& rhs) {
        return Base::cast() = (Base::cast() / rhs);
      }
  };

  template<typename T_InnerReal, typename T_Tape>
  struct ExpressionTraits::ActiveResultImpl<std::complex<T_InnerReal>, T_Tape, false> {

      using InnerReal = CODI_DD(T_InnerReal, CODI_ANY);
      using Tape = CODI_DD(T_Tape, CODI_ANY);

      using InnerActiveResult = ExpressionTraits::ActiveResult<InnerReal, Tape, false>;

      /// The resulting active type of an expression.
#if CODI_SpecializeStdComplex
      using ActiveResult = std::complex<InnerActiveResult>;
#else
      using ActiveResult = ActiveComplex<InnerActiveResult>;
#endif
  };

  template<typename T_InnerReal, typename T_Tape>
  struct ExpressionTraits::ActiveResultImpl<std::complex<T_InnerReal>, T_Tape, true> {

      using InnerReal = CODI_DD(T_InnerReal, CODI_ANY);
      using Tape = CODI_DD(T_Tape, CODI_ANY);

      using InnerActiveResult = ExpressionTraits::ActiveResult<InnerReal, Tape, true>;

      using ActiveResult = StaticAggregatedExpressionType<std::complex<InnerReal>, InnerActiveResult>;
  };
}

#if CODI_SpecializeStdComplex
namespace std {

  /*******************************************************************************/
  /// @name Specialization of std::complex for active types.
  /// @{

  template<typename T_Tape>
  class complex<codi::ActiveType<T_Tape>> :
      public codi::ActiveComplex<codi::ActiveType<T_Tape>, complex<codi::ActiveType<T_Tape>>> {

    public:

      using Tape = CODI_DD(T_Tape, CODI_T(codi::FullTapeInterface<double, double, int, codi::EmptyPosition>));

      using Real = complex<typename Tape::Real>;
      using InnerActive = codi::ActiveType<Tape>;
      using InnerReal = typename InnerActive::Real;

      using Base = codi::ActiveComplex<codi::ActiveType<Tape>, complex<codi::ActiveType<T_Tape>>>;

      using Base::Base;
      complex(complex const& value) : Base(value) {}

      using Base::operator =;
      complex& operator =(complex const& value) {
        Base::store(value);

        return *this;
      }

      /*******************************************************************************/
      /// Implementation of ExpressionInterface

      //using StoreAs = complex const&;
  };

  /// @}
  /*******************************************************************************/
  /// @name Binary  operators and standard math library binary functions
  /// @{

#define FUNCTION operator+
#define OPERATION_LOGIC codi::OperationAdd
#include "binaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION operator-
#define OPERATION_LOGIC codi::OperationSubstract
#include "binaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION operator*
#define OPERATION_LOGIC codi::OperationMultiply
#include "binaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION operator/
#define OPERATION_LOGIC codi::OperationDivide
#include "binaryComplexToComplexStdSpecialization.tpp"

  // polar needs to be created in the codi namespace.

#define FUNCTION pow
#define OPERATION_LOGIC codi::OperationPow
#include "binaryComplexToComplexStdSpecialization.tpp"

  /// @}
  /*******************************************************************************/
  /// @name Unary operators and standard math library unary functions
  /// @{


  /// Function overload for operator +
  template<typename Tape>
  CODI_INLINE complex<codi::ActiveType<Tape>> const& operator+(complex<codi::ActiveType<Tape>> const& arg) {
    return arg;
  }

#define FUNCTION operator-
#define OPERATION_LOGIC codi::OperationUnaryMinus
#include "unaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION abs
#define OPERATION_LOGIC codi::OperationComplexAbs
#include "unaryComplexToRealStdSpecialization.tpp"

#define FUNCTION acos
#define OPERATION_LOGIC codi::OperationAcos
#include "unaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION acosh
#define OPERATION_LOGIC codi::OperationAcosh
#include "unaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION arg
#define OPERATION_LOGIC codi::OperationComplexArg
#include "unaryComplexToRealStdSpecialization.tpp"

#define FUNCTION asin
#define OPERATION_LOGIC codi::OperationAsin
#include "unaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION asinh
#define OPERATION_LOGIC codi::OperationAsinh
#include "unaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION atan
#define OPERATION_LOGIC codi::OperationAtan
#include "unaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION atanh
#define OPERATION_LOGIC codi::OperationAtanh
#include "unaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION conj
#define OPERATION_LOGIC codi::OperationComplexConj
#include "unaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION cos
#define OPERATION_LOGIC codi::OperationCos
#include "unaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION cosh
#define OPERATION_LOGIC codi::OperationCosh
#include "unaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION exp
#define OPERATION_LOGIC codi::OperationExp
#include "unaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION imag
#define OPERATION_LOGIC codi::OperationComplexImag
#include "unaryComplexToRealStdSpecialization.tpp"

#define FUNCTION log
#define OPERATION_LOGIC codi::OperationLog
#include "unaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION log10
#define OPERATION_LOGIC codi::OperationLog10
#include "unaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION norm
#define OPERATION_LOGIC codi::OperationComplexNorm
#include "unaryComplexToRealStdSpecialization.tpp"

#define FUNCTION proj
#define OPERATION_LOGIC codi::OperationComplexProj
#include "unaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION real
#define OPERATION_LOGIC codi::OperationComplexReal
#include "unaryComplexToRealStdSpecialization.tpp"

#define FUNCTION sin
#define OPERATION_LOGIC codi::OperationSin
#include "unaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION sinh
#define OPERATION_LOGIC codi::OperationSinh
#include "unaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION sqrt
#define OPERATION_LOGIC codi::OperationSqrt
#include "unaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION tan
#define OPERATION_LOGIC codi::OperationTan
#include "unaryComplexToComplexStdSpecialization.tpp"

#define FUNCTION tanh
#define OPERATION_LOGIC codi::OperationTanh
#include "unaryComplexToComplexStdSpecialization.tpp"

  /// @}
}
#endif
