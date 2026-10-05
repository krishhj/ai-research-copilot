from fastapi import APIRouter, Depends

from app.core.auth import AuthenticatedUser, get_current_user


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.get("/me")
def get_me(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, str | None]:
    """Return the currently authenticated user's safe identity details."""
    return {
        "id": str(current_user.id),
        "email": current_user.email,
    }