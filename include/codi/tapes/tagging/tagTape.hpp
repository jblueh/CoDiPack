/*
 * CoDiPack, a Code Differentiation Package
 *
 * Copyright (C) 2015-2022 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (codi@scicomp.uni-kl.de)
 *
 * Lead developers: Max Sagebaum, Johannes Blühdorn (SciComp, TU Kaiserslautern)
 *
 * This file is part of CoDiPack (http://www.scicomp.uni-kl.de/software/codi).
 *
 * CoDiPack is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * CoDiPack is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU
 * General Public License along with CoDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * For other licensing options please contact us.
 *
 * Authors:
 *  - SciComp, TU Kaiserslautern:
 *    - Max Sagebaum
 *    - Johannes Blühdorn
 *    - Former members:
 *      - Tim Albring
 */
#pragma once

#include "../../expressions/logic/helpers/forEachLeafLogic.hpp"
#include "../../misc/enumBitset.hpp"
#include "../indices/indexManagerInterface.hpp"
#include "../interfaces/fullTapeInterface.hpp"
#include "../misc/adjointVectorAccess.hpp"
#include "tagData.hpp"

/** \copydoc codi::Namespace */
namespace codi {

  /// Helper class for statement validation.
  template<typename Tag>
  struct ValidationIndicator {
      bool isActive;     ///< true if an active rhs is detected. tag != 0
      bool hasError;     ///< true if an error is detected.
      bool hasTagError;  ///< true if a tag not the current required tag.
      bool hasUseError;  ///< true if a value is used in the wrong way.
      Tag errorTag;      ///< Value of the wrong tag.

      /// Constructor.
      ValidationIndicator() : isActive(false), hasError(false), hasTagError(false), hasUseError(false), errorTag() {}
  };

  /**
   * @brief Tape for tagging variables and find errors in the AD workflow.
   *
   * @tparam T_Real  The computation type of a tape, usually chosen as ActiveType::Real.
   * @tparam T_tag   The type of the tag, usually int.
   */
  template<typename T_Real, typename T_Tag>
  struct TagTape : public FullTapeInterface<T_Real, T_Real, TagData<T_Tag>, EmptyPosition> {
      using Real = CODI_DD(T_Real, double);  ///< See TagTape.
      using Tag = CODI_DD(T_Tag, int);       ///< See TagTape.

      /// Required definition for event system.
      struct TapeTypes {
          /// Required definition for event system.
          struct IndexManager {
              /// Required definition for event system.
              using Index = int;
          };
      };

      using Gradient = Real;            ///< See TapeTypesInterface.
      using Identifier = TagData<Tag>;  ///< See TapeTypesInterface.
      using Position = EmptyPosition;   ///< See TapeTypesInterface.

      using PassiveReal = RealTraits::PassiveReal<Real>;  ///< Basic computation type.

      /// Callback for a change in a lhs value.
      using TagLhsChangeErrorCallback = void (*)(Real const& currentValue, Real const& newValue, void* userData);

      /// Callback for a tag error.
      using TagErrorCallback = void (*)(Tag const& correctTag, Tag const& wrongTag, bool tagError, bool useError,
                                        void* userData);

    private:

      Tag curTag;   ///< Current tag for new values.
      bool active;  ///< Tape activity.

      Real tempPrimal;        ///< Temporary for primal values.
      Gradient tempGradient;  ///< Temporary for gradient values.

      TagLhsChangeErrorCallback tagLhsChangeErrorCallback;  ///< User defined callback for lhs value errors.
      void* tagChangeErrorUserData;                         ///< User data in call to callback for lhs value errors.

      TagErrorCallback tagErrorCallback;  ///< User defined callback for tag errors.
      void* tagErrorUserData;             ///< User data in call to callback for tag errors.

      std::set<TapeParameters> parameters;  /// Temporary for tape parameters.

      bool preaccumulationHandling;  ///< Parameter to enable disable preaccumulation handling.
      Tag preaccumulationTag;        ///< Tag used for preaccumulation specialized handling.

    public:

      /// Constructor.
      TagTape()
          : curTag(),
            active(),
            tempPrimal(),
            tempGradient(),
            tagLhsChangeErrorCallback(defaultTagLhsChangeErrorCallback),
            tagChangeErrorUserData(nullptr),
            tagErrorCallback(defaultTagErrorCallback),
            tagErrorUserData(nullptr),
            parameters(),
            preaccumulationHandling(true),
            preaccumulationTag(1337) {}

      /*******************************************************************************/
      /// @name CustomAdjointVectorEvaluationTapeInterface interface implementation
      /// @{

      /// Do nothing.
      template<typename Adjoint>
      void evaluate(Position const& start, Position const& end, Adjoint* data) {
        CODI_UNUSED(start, end, data);
      }

      /// Do nothing.
      template<typename Adjoint>
      void evaluateForward(Position const& start, Position const& end, Adjoint* data) {
        CODI_UNUSED(start, end, data);
      }

      /// @}
      /*******************************************************************************/
      /// @name DataManagementTapeInterface interface implementation
      /// @{

      /// Do nothing.
      void writeToFile(std::string const& filename) const {
        CODI_UNUSED(filename);
      }

      /// Do nothing.
      void readFromFile(std::string const& filename) {
        CODI_UNUSED(filename);
      }

      /// Do nothing.
      void deleteData() {}

      /// Do nothing.
      std::set<TapeParameters> const& getAvailableParameters() const {
        return parameters;
      }

      /// Do nothing.
      size_t getParameter(TapeParameters parameter) const {
        return 0;
      }

      /// Do nothing.
      bool hasParameter(TapeParameters parameter) const {
        return false;
      }

      /// Do nothing.
      void setParameter(TapeParameters parameter, size_t value) {
        CODI_UNUSED(parameter, value);
      }

      /// Do nothing.
      VectorAccessInterface<Real, Identifier>* createVectorAccess() {
        return new AdjointVectorAccess<Real, Identifier, Real>(nullptr);
      }

      /// Do nothing.
      template<typename Adjoint>
      VectorAccessInterface<Real, Identifier>* createVectorAccessCustomAdjoints(Adjoint* data) {
        CODI_UNUSED(data);
        return new AdjointVectorAccess<Real, Identifier, Real>(nullptr);
      }

      /// Do nothing.
      void deleteVectorAccess(VectorAccessInterface<Real, Identifier>* access) {
        delete access;
      }

      /// Do nothing.
      void swap(TagTape& other) {
        std::swap(curTag, other.curTag);
      }
      void resetHard() {}            ///< Do nothing.
      void deleteAdjointVector() {}  ///< Do nothing.

      /// @}
      /*******************************************************************************/
      /// @name ExternalFunctionTapeInterface interface implementation
      /// @{

      /// Verifies tag properties.
      template<typename Lhs>
      Real registerExternalFunctionOutput(LhsExpressionInterface<Real, Gradient, TagTape, Lhs>& value) {
        registerInput(value);

        return Real();
      }

      /// Do nothing.
      void pushExternalFunction(ExternalFunction<TagTape> const& extFunc) {
        CODI_UNUSED(extFunc);
      }

      /// @}
      /*******************************************************************************/
      /// @name ForwardEvaluationTapeInterface interface implementation
      /// @{

      /// Do nothing.
      void evaluateForward(Position const& start, Position const& end) {
        CODI_UNUSED(start, end);
      }

      /// Do nothing.
      void evaluateForward() {}

      /// @}
      /*******************************************************************************/
      /// @name GradientAccessTapeInterface interface implementation
      /// @{

      /// Verify tag.
      void setGradient(Identifier const& identifier, Gradient const& gradient) {
        CODI_UNUSED(gradient);

        verifyTag(identifier.tag);
      }

      /// Verify tag.
      Gradient const& getGradient(Identifier const& identifier) const {
        verifyTag(identifier.tag);

        return tempGradient;
      }

      /// Verify tag.
      Gradient& gradient(Identifier const& identifier) {
        verifyTag(identifier.tag);

        return tempGradient;
      }

      /// Verify tag.
      Gradient const& gradient(Identifier const& identifier) const {
        verifyTag(identifier.tag);

        return tempGradient;
      }

      /// @}
      /*******************************************************************************/
      /// @name IdentifierInformationTapeInterface interface implementation
      /// @{

      /// Behave as linear index handler.
      static bool constexpr LinearIndexHandling = true;

      /// Zero tag.
      Identifier getPassiveIndex() const {
        return Identifier(0);
      }

      /// -1 tag.
      Identifier getInvalidIndex() const {
        return Identifier(-1);
      }

      /// Verify tag.
      bool isIdentifierActive(Identifier const& index) const {
        verifyTag(index.tag);

        return index.tag != 0;
      }

      /// Set tag to passive.
      template<typename Lhs>
      void deactivateValue(LhsExpressionInterface<Real, Gradient, TagTape, Lhs>& value) {
        value.getIdentifier() = getPassiveIndex();
      }

      /// @}
      /*******************************************************************************/
      /// @name InternalStatementRecordingTapeInterface interface implementation
      /// @{

      /// Do not allow Jacobian optimization.
      static bool constexpr AllowJacobianOptimization = false;

      /// Do nothing.
      template<typename Real>
      void initIdentifier(Real& value, Identifier& identifier) {
        CODI_UNUSED(value);
        identifier = Identifier();
      }

      /// Do nothing.
      template<typename Real>
      void destroyIdentifier(Real& value, Identifier& identifier) {
        CODI_UNUSED(value, identifier);
      }

    protected:

      /// Looks at the tags for the expression.
      struct ValidateTags : public ForEachLeafLogic<ValidateTags> {
        public:

          /// \copydoc codi::ForEachLeafLogic::handleActive
          template<typename Node>
          CODI_INLINE void handleActive(Node const& node, ValidationIndicator<Tag>& vi, TagTape& tape) {
            Identifier tagData = node.getIdentifier();
            tape.verifyTag(vi, tagData.tag);
            tape.verifyProperties(vi, tagData.properties);
          }
      };

    public:

      /// Verify all tags of the rhs and the lhs properties.
      template<typename Lhs, typename Rhs>
      CODI_INLINE void store(LhsExpressionInterface<Real, Gradient, TagTape, Lhs>& lhs,
                             ExpressionInterface<Real, Rhs> const& rhs) {
        ValidateTags validate;
        ValidationIndicator<Tag> vi;

        validate.eval(rhs, vi, *this);

        checkLhsError(lhs, rhs.cast().getValue());

        handleError(vi);

        if (vi.isActive) {
          setTag(lhs.cast().getIdentifier().tag);
        } else {
          resetTag(lhs.cast().getIdentifier().tag);
        }
        lhs.cast().value() = rhs.cast().getValue();
      }

      /// Verify all tags of the rhs and the lhs properties.
      template<typename Lhs, typename Rhs>
      CODI_INLINE void store(LhsExpressionInterface<Real, Gradient, TagTape, Lhs>& lhs,
                             LhsExpressionInterface<Real, Gradient, TagTape, Rhs> const& rhs) {
        store<Lhs, Rhs>(lhs, static_cast<ExpressionInterface<Real, Rhs> const&>(rhs));
      }

      /// Verify the lhs properties.
      template<typename Lhs>
      CODI_INLINE void store(LhsExpressionInterface<Real, Gradient, TagTape, Lhs>& lhs, PassiveReal const& rhs) {
        checkLhsError(lhs, rhs);

        resetTag(lhs.cast().getIdentifier().tag);

        lhs.cast().value() = rhs;
      }

      /// @}
      /*******************************************************************************/
      /// @name ManualStatementPushTapeInterface interface implementation
      /// @{

      /// Do nothing.
      void pushJacobiManual(Real const& jacobian, Real const& value, Identifier const& index) {
        CODI_UNUSED(jacobian, value, index);
      }

      /// Do nothing.
      void storeManual(Real const& lhsValue, Identifier& lhsIndex, Config::ArgumentSize const& size) {
        CODI_UNUSED(lhsValue, size);

        checkLhsError(lhsValue, lhsIndex, lhsValue);
        setTag(lhsIndex.tag);
      }

      /// @}
      /*******************************************************************************/
      /// @name PositionalEvaluationTapeInterface interface implementation
      /// @{

      /// Do nothing.
      void evaluate(Position const& start, Position const& end) {
        CODI_UNUSED(start, end);
      }

      /// Do nothing.
      void clearAdjoints(Position const& start, Position const& end) {
        CODI_UNUSED(start, end);
      }

      /// Do nothing.
      Position getPosition() const {
        return Position();
      }

      /// Do nothing.
      Position getZeroPosition() const {
        return Position();
      }

      /// Do nothing.
      void resetTo(Position const& pos, bool resetAdjoints = true) {
        CODI_UNUSED(pos, resetAdjoints);
      }

      /// @}
      /*******************************************************************************/
      /// @name PreaccumulationEvaluationTapeInterface interface implementation
      /// @{

      /// Do nothing.
      void evaluateKeepState(Position const& start, Position const& end) {
        CODI_UNUSED(start, end);
      }
      /// Do nothing.
      void evaluateForwardKeepState(Position const& start, Position const& end) {
        CODI_UNUSED(start, end);
      }

      /// @}
      /*******************************************************************************/
      /// @name PrimalEvaluationTapeInterface interface implementation
      /// @{

      static bool constexpr HasPrimalValues = false;        ///< No primal values.
      static bool constexpr RequiresPrimalRestore = false;  ///< No primal values.

      /// Do nothing.
      void evaluatePrimal(Position const& start, Position const& end) {
        CODI_UNUSED(start, end);
      }

      /// Do nothing.
      void evaluatePrimal() {}

      /// Do nothing.
      void setPrimal(Identifier const& identifier, Real const& gradient) {
        CODI_UNUSED(identifier, gradient);
      }

      /// Do nothing.
      Real const& getPrimal(Identifier const& identifier) const {
        CODI_UNUSED(identifier);
        return tempPrimal;
      }

      /// Do nothing.
      Real& primal(Identifier const& identifier) {
        CODI_UNUSED(identifier);
        return tempPrimal;
      }

      /// Do nothing.
      Real const& primal(Identifier const& identifier) const {
        CODI_UNUSED(identifier);
        return tempPrimal;
      }

      /// Do nothing.
      void revertPrimals(Position const& pos) {
        CODI_UNUSED(pos);
      }

      /// @}
      /*******************************************************************************/
      /// @name ReverseTapeInterface interface implementation
      /// @{

      /// Verify value properties.
      template<typename Lhs>
      void registerInput(LhsExpressionInterface<Real, Gradient, TagTape, Lhs>& value) {
        setTag(value.cast().getIdentifier().tag);
        verifyRegisterValue(value, value.cast().getIdentifier());  // verification is mainly for the properties
      }

      /// Verify tag.
      template<typename Lhs>
      void registerOutput(LhsExpressionInterface<Real, Gradient, TagTape, Lhs>& value) {
        verifyRegisterValue(value, value.cast().getIdentifier());
      }

      void setActive() {
        active = true;
      }  ///< Set tape to active.
      void setPassive() {
        active = false;
      }  ///< Set tape to passive.
      bool isActive() const {
        return active;
      }  ///< Check if tape is active.

      void evaluate() {}  ///< Do nothing.

      /// Do nothing.
      void clearAdjoints() {}

      /// Do nothing.
      void reset(bool resetAdjoints = true) {
        CODI_UNUSED(resetAdjoints);
      }

      /// Do nothing.
      template<typename Stream = std::ostream>
      void printStatistics(Stream& out = std::cout) const {
        CODI_UNUSED(out);
      }

      /// Do nothing.
      template<typename Stream = std::ostream>
      void printTableHeader(Stream& out = std::cout) const {
        CODI_UNUSED(out);
      }

      /// Do nothing.
      template<typename Stream = std::ostream>
      void printTableRow(Stream& out = std::cout) const {
        CODI_UNUSED(out);
      }

      /// Do nothing.
      TapeValues getTapeValues() const {
        return TapeValues("TagTape");
      }

      /// @}
      /*******************************************************************************/
      /// @name Tagging specific functions.
      /// @{

      /// Set the current tag of the tape.
      void setCurTag(const Tag& tag) {
        this->curTag = tag;
      }

      /// Get the current tag of the tape.
      Tag getCurTag() {
        return this->curTag;
      }

      /// Get tag of a CoDiPack active type.
      template<typename Lhs>
      Tag getTag(LhsExpressionInterface<Real, Gradient, TagTape, Lhs>& value) {
        return value.cast().getIdentifier().tag;
      }

      /// Set tag on a CoDiPack active type.
      template<typename Lhs>
      void setTagOnVariable(LhsExpressionInterface<Real, Gradient, TagTape, Lhs>& value) {
        value.cast().getIdentifier().tag = this->curTag;
      }

      /// Clear tag on a CoDiPack active type.
      template<typename Lhs>
      void clearTagOnVariable(LhsExpressionInterface<Real, Gradient, TagTape, Lhs>& value) {
        value.cast().getIdentifier().tag = Tag();
      }

      /// Clear properties on a CoDiPack active type.
      template<typename Lhs>
      void clearTagProperties(LhsExpressionInterface<Real, Gradient, TagTape, Lhs>& value) {
        value.cast().getIdentifier().properties.reset();
      }

      /// Set properties on a CoDiPack active type.
      template<typename Lhs>
      void setTagProperty(LhsExpressionInterface<Real, Gradient, TagTape, Lhs>& value, TagFlags flag) {
        value.cast().getIdentifier().properties.set(flag);
      }

      /// Check properties on a CoDiPack active type.
      template<typename Lhs>
      bool hasTagProperty(LhsExpressionInterface<Real, Gradient, TagTape, Lhs>& value, TagFlags flag) {
        return value.cast().getIdentifier().properties.test(flag);
      }

      /// Set the callback and user data for a lhs error.
      void setTagLhsChangeErrorCallback(TagLhsChangeErrorCallback const& callback, void* userData) {
        tagLhsChangeErrorCallback = callback;
        tagChangeErrorUserData = userData;
      }

      /// Set the callback and user data for a tag error.
      void setTagErrorCallback(TagErrorCallback const& callback, void* userData) {
        tagErrorCallback = callback;
        tagErrorUserData = userData;
      }

      /// Enable or disable specialized handling for preaccumulation. Default: true
      ///
      /// Uses a special tag to sanetize preaccumulation regions.
      CODI_INLINE void setPreaccumulationHandlingEnabled(bool enabled) {
       preaccumulationHandling = enabled;
      }

      /// Set the special tag for preaccumulation regions. See setPreaccumulationHandlingEnabled().
      CODI_INLINE void setPreaccumulationHandlingTag(Tag tag) {
       preaccumulationTag = tag;
      }

      /// If handling for preaccumulation is enabled.
      CODI_INLINE bool isPreaccumulationHandlingEnabled() {
        return preaccumulationHandling;
      }

      /// The special tag for preaccumulation.
      CODI_INLINE Tag getPreaccumulationHandlingTag() {
        return preaccumulationTag;
      }

    private:

      /// Checks if the tag is correct.
      CODI_INLINE void verifyTag(ValidationIndicator<Tag>& vi, Tag const& tag) const {
        if (0 != tag) {
          vi.isActive = true;
          if (tag != curTag) {
            vi.hasError = true;
            vi.hasTagError = true;
            vi.errorTag = tag;
          }
        }
      }

      /// Checks if the tag is correct and creates an error.
      CODI_INLINE void verifyTag(Tag const& tag) const {
        ValidationIndicator<Tag> vi;

        verifyTag(vi, tag);
        handleError(vi);
      }

      /// Checks if the tag is correct.
      CODI_INLINE void verifyProperties(ValidationIndicator<Tag>& vi, const EnumBitset<TagFlags>& properties) const {
        if (properties.test(TagFlags::DoNotUse)) {
          vi.hasError = true;
          vi.hasUseError = true;
        }
      }

      /// Default callback for TagLhsChangeErrorCallback.
      static void defaultTagLhsChangeErrorCallback(Real const& currentValue, Real const& newValue, void* userData) {
        CODI_UNUSED(userData);

        std::cerr << "Wrong tag use detected '" << currentValue << "'' is set to '" << newValue << "'." << std::endl;
      }

      /// Default callback for TagErrorCallback.
      static void defaultTagErrorCallback(Tag const& correctTag, Tag const& wrongTag, bool tagError, bool useError,
                                          void* userData) {
        CODI_UNUSED(userData);

        // output default warning if no handle is defined.
        if (useError) {
          std::cerr << "Wrong variable use detected." << std::endl;
        }
        if (tagError) {
          std::cerr << "Wrong tag detected '" << wrongTag << "' should be '" << correctTag << "'." << std::endl;
        }
      }

      /// Check if the lhs value is changed.
      CODI_INLINE void checkLhsError(Real& lhsValue, Identifier& lhsIdentifier, const Real& rhs) const {
        if (lhsIdentifier.properties.test(TagFlags::DoNotChange)) {
          if(lhsValue != rhs) {
            tagLhsChangeErrorCallback(lhsValue, rhs, tagChangeErrorUserData);
          }
        }
      }

      /// Check if the lhs value is changed.
      template<typename Lhs>
      CODI_INLINE void checkLhsError(LhsExpressionInterface<Real, Gradient, TagTape, Lhs>& lhs, const Real& rhs) const {
        checkLhsError(lhs.cast().value(), lhs.cast().getIdentifier(), rhs);
      }

      /// Call tag error callback.
      CODI_INLINE void handleError(ValidationIndicator<Tag>& vi) const {
        if (vi.hasError) {
          tagErrorCallback(curTag, vi.errorTag, vi.hasTagError, vi.hasUseError, tagErrorUserData);
        }
      }

      /// Verify tag, properties and lhs error.
      template<typename Lhs>
      CODI_INLINE void verifyRegisterValue(LhsExpressionInterface<Real, Gradient, TagTape, Lhs>& value,
                                           const Identifier& tag) {
        ValidationIndicator<Tag> vi;

        verifyTag(vi, tag.tag);
        verifyProperties(vi, tag.properties);
        handleError(vi);

        checkLhsError(value, value.cast().getValue());
      }

      /// Set tag on value.
      CODI_INLINE void setTag(Tag& tag) const {
        tag = curTag;
      }

      /// Reset tag on value.
      CODI_INLINE void resetTag(Tag& tag) const {
        tag = Tag();
      }

      /// @}
  };
}
