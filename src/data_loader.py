import pandas as pd
from typing import List, Dict, Any
from .config import settings

class PropertyDataLoader:
    def __init__(self):
        self.properties_file = settings.properties_file
        self.properties_df = None
        self._load_data()

    def _load_data(self) -> None:
        """Load property data from CSV file."""
        try:
            self.properties_df = pd.read_csv(self.properties_file)
            # Convert string lists to actual lists
            self.properties_df['amenities'] = self.properties_df['amenities'].apply(
                lambda x: [item.strip() for item in x.split(',')]
            )
            self.properties_df['available_months'] = self.properties_df['available_months'].apply(
                lambda x: [month.strip() for month in x.split(',')]
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Properties file not found at {self.properties_file}")
        except Exception as e:
            raise Exception(f"Error loading properties data: {str(e)}")

    def get_all_properties(self) -> List[Dict[str, Any]]:
        """Get all properties as a list of dictionaries."""
        return self.properties_df.to_dict('records')

    def get_available_properties(self) -> List[Dict[str, Any]]:
        """Get only available properties."""
        return self.properties_df[self.properties_df['status'] == 'available'].to_dict('records')

    def get_property_by_id(self, property_id: int) -> Dict[str, Any]:
        """Get a specific property by ID."""
        property_data = self.properties_df[self.properties_df['property_id'] == property_id]
        if property_data.empty:
            raise ValueError(f"Property with ID {property_id} not found")
        return property_data.iloc[0].to_dict()

    def get_properties_by_location(self, location: str) -> List[Dict[str, Any]]:
        """Get properties in a specific location."""
        return self.properties_df[self.properties_df['location'].str.lower() == location.lower()].to_dict('records')

    def get_properties_by_price_range(self, min_price: float, max_price: float) -> List[Dict[str, Any]]:
        """Get properties within a price range."""
        return self.properties_df[
            (self.properties_df['price'] >= min_price) & 
            (self.properties_df['price'] <= max_price)
        ].to_dict('records')

    def get_properties_by_amenity(self, amenity: str) -> List[Dict[str, Any]]:
        """Get properties that have a specific amenity."""
        return self.properties_df[
            self.properties_df['amenities'].apply(lambda x: amenity.lower() in [a.lower() for a in x])
        ].to_dict('records') 