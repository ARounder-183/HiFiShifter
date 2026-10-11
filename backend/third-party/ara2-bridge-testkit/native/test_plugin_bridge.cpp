#include "cpp_interop.h"

#include "TestPlugIn/ARATestDocumentController.h"

#include <cstring>
#include <exception>

namespace
{
void resetResult (Ara2NativeResult* result) noexcept
{
    if (result != nullptr)
        std::memset (result, 0, sizeof (*result));
}

void setDiagnostic (Ara2NativeResult* result, const char* message) noexcept
{
    if (result == nullptr)
        return;
    std::strncpy (result->diagnostic, message, sizeof (result->diagnostic) - 1U);
    result->diagnostic[sizeof (result->diagnostic) - 1U] = '\0';
}
} // namespace

extern "C" const Ara2ARAFactory* ara2_cpp_test_plugin_factory (Ara2NativeResult* result)
{
    resetResult (result);
    if (result == nullptr)
        return nullptr;

    try
    {
        const auto* factory { ARATestDocumentController::getARAFactory () };
        if (factory == nullptr)
        {
            result->status = kAra2NativeAraFailure;
            setDiagnostic (result, "Celemony TestPlugIn returned a null ARA factory");
            return nullptr;
        }
        result->status = kAra2NativeOk;
        result->generation = factory->highestSupportedApiGeneration;
        return factory;
    }
    catch (const std::exception& error)
    {
        result->status = kAra2NativeException;
        setDiagnostic (result, error.what ());
    }
    catch (...)
    {
        result->status = kAra2NativeException;
        setDiagnostic (result, "unknown C++ exception while acquiring TestPlugIn factory");
    }
    return nullptr;
}
