{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block methods %}
   .. automethod:: __init__

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
      :recursive:
   {% for item in methods %}
   {% if item != "__init__" %}
   {%- if item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {%- endif %}  
   {%- endif %}   
   {%- endfor %}
   {% endif %}
   {% endblock %}
